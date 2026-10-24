"""Fetch selected ENDF/B-VII.1 archive members with pinned source checksums."""
from urllib.request import Request, urlopen
import hashlib
import io
import zipfile
from runtime.io import sha, save_json

BASE = "https://www.nndc.bnl.gov/endf-b7.1/zips/"
MAX_REQUEST_BYTES = 32 * 1024**2


class ArchiveRanges(io.RawIOBase):
    """Seekable HTTP ranges for ZipFile; refuse accidental full-library reads."""
    def __init__(self, url):
        """Open a seekable HTTP-range view of a remote release archive.

        Parameters
        ----------
        url : str
            Archive URL whose server supports HEAD and byte-range GET requests.
        """
        self.url, self.position, self.transferred = url, 0, 0
        with urlopen(Request(url, method='HEAD'), timeout=30) as response:
            self.length = int(response.headers['Content-Length'])
            self.etag = response.headers.get('ETag')
        self.etags = {self.etag}

    def seekable(self):
        """Report that the remote archive supports seeking.

        Returns
        -------
        bool
            Always True.
        """
        return True

    def readable(self):
        """Report that the remote archive supports reading.

        Returns
        -------
        bool
            Always True.
        """
        return True

    def tell(self):
        """Return the current byte offset within the archive.

        Returns
        -------
        int
            Zero-based logical stream position in bytes.
        """
        return self.position

    def seek(self, offset, whence=0):
        """Move the logical archive position without downloading data.

        Parameters
        ----------
        offset : int
            Byte displacement from the selected origin.
        whence : {0, 1, 2}, optional
            Origin: beginning, current position or end of archive.

        Returns
        -------
        int
            New logical byte position.

        Raises
        ------
        ValueError
            The requested position is negative.
        """
        position = offset+({0: 0, 1: self.position, 2: self.length}[whence])
        if position < 0:
            raise ValueError('Negative archive offset')
        self.position = position
        return position

    def read(self, size=-1):
        """Fetch a bounded byte range while checking archive identity.

        Parameters
        ----------
        size : int, optional
            Requested byte count; a negative value requests the remainder subject to
            MAX_REQUEST_BYTES.

        Returns
        -------
        bytes
            Requested archive bytes, or empty bytes at end of stream.

        Raises
        ------
        ValueError
            The request exceeds the allowed range-download size.
        RuntimeError
            The server ignores ranges, changes archive identity or returns incomplete data.
        """
        count = self.length-self.position if size < 0 else min(size, self.length-self.position)
        if count <= 0:
            return b''
        if count > MAX_REQUEST_BYTES:
            raise ValueError('Refusing a bulk archive read')
        end = self.position+count-1
        request = Request(self.url, headers={'Range': f'bytes={self.position}-{end}', 'Accept-Encoding': 'identity'})
        with urlopen(request, timeout=60) as response:
            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {self.position}-{end}/{self.length}':
                raise RuntimeError('Archive server did not honor the requested byte range')
            actual_etag = response.headers.get('ETag')
            # NNDC backends report different modification times for the same
            # release. Weak ETags cannot establish byte identity; member CRCs
            # and retained SHA-256 values provide the content checks here.
            if self.etag and not self.etag.startswith('W/') and actual_etag != self.etag:
                raise RuntimeError('Archive changed during selection')
            self.etags.add(actual_etag)
            data = response.read(count+1)
        if len(data) != count:
            raise RuntimeError('Incomplete archive range')
        self.position += count
        self.transferred += count
        return data


def fetch_sources(record, *, members, sources, receipt):
    """Fetch selected ENDF archive members and verify their declared content.

    Parameters
    ----------
    record : dict
        Mutable source manifest containing expected SHA-256 values where supplied.
    members : dict
        Archive-category mappings from nuclide/scattering names to ZIP member paths.
    sources : pathlib.Path
        Destination directory for selected ENDF evaluations.
    receipt : pathlib.Path
        Source-manifest path updated as members are verified.

    Raises
    ------
    RuntimeError
        Downloaded or existing files differ from their declared source identities.
    """
    sources.mkdir(parents=True, exist_ok=True)
    for library, selected in members.items():
        needed = [name for name in selected if not (sources/f'{name}.endf').exists()]
        if needed:
            url = BASE+f'ENDF-B-VII.1-{library}.zip'
            with ArchiveRanges(url) as remote, zipfile.ZipFile(remote) as archive:
                for name in needed:
                    member = selected[name]
                    info = archive.getinfo(member)
                    data = archive.read(member)  # ZipFile verifies the member CRC.
                    expected = record['sources'].get(name, {}).get('sha256')
                    if expected is not None and hashlib.sha256(data).hexdigest() != expected:
                        raise RuntimeError(f'Download differs from the pinned evaluation: {name}')
                    path = sources/f'{name}.endf'
                    path.write_bytes(data)
                    record['sources'][name] = {'url': url, 'member': member,
                        'archive_etags_observed': sorted(v for v in remote.etags if v is not None),
                        'archive_bytes': remote.length, 'zip_crc32': info.CRC, 'bytes': len(data),
                        'sha256': sha(path)}
                    save_json(receipt, record)
                record.setdefault('download_bytes', {})[library] = remote.transferred
        for name in selected:
            if name not in record['sources']:
                raise RuntimeError(f'Missing provenance for existing source {name}')
            if sha(sources/f'{name}.endf') != record['sources'][name]['sha256']:
                raise RuntimeError(f'Existing source differs from its pinned checksum: {name}')
        save_json(receipt, record)
