# Reactor design questions

1. **Hydrogen embrittlement.** How does organic coolant chemistry affect hydrogen uptake and embrittlement in the fuel cladding and pressure tubes?

2. **Minimum coolant velocity.** What explains the low-velocity limits reported for WR-1? Were both power and flow reduced in the tests, or can lower velocity increase fouling even at similar surface temperatures?

3. **Coolant pressure.** Why was the organic coolant pressurized if this was described as a low-pressure reactor?

4. **Pressure tubes and calandria tubes.** Why are both needed? How does the gap between them limit heat transfer to the moderator?

5. **Fuel failure.** What happens when the cladding fails and organic coolant comes into contact with the fuel?

6. **On-power refuelling.** How does refuelling during operation work? Does each channel have its own shutoff valve, and how is the pressure boundary maintained during fuel exchange?

7. **Starting with degraded coolant.** Did WR-1 actually start with degraded coolant, or was this only proposed? How is the operating composition established and maintained as irradiation continues? Review the available coolant reports.

8. **Component ageing and replacement.** How much does low temperature limit ageing of the calandria vessel and tubes, and what irradiation damage or creep remains relevant? Can the pressure tubes and calandria tubes be replaced? Examine how this is done in water-cooled CANDU reactors.

9. **Circular fuel bundles and procurement.** Why use a circular bundle? What are the manufacturing and pressure-boundary advantages, and what thermal-hydraulic complications arise? Where could these bundles be procured, given that they differ from PWR fuel assemblies?

10. **Moderator-level control.** How did WR-1 achieve fine reactivity control without conventional control rods? Was moderator-level control adequate for the required operating manoeuvres?

11. **Grey rods and shutdown.** Would grey rods simplify routine control or manufacture? Could moderator dumping then be reserved for scrams? How quickly does a dump introduce sufficient negative reactivity?

12. **Partial moderator dumping and heat removal.** Dumping the moderator changes the cooling of the calandria tubes, so their temperature cannot simply be assumed constant when assessing heat transfer from the fuel. At what moderator level is the reactor sufficiently subcritical? Could a partial dump provide the required shutdown margin while retaining moderator for heat removal?

13. **Comparison with water-cooled CANDUs.** Which features are shared with operating CANDU designs? Did any use moderator-level control as their primary regulating method, and how did their shutdown systems differ?

14. **Fuel geometry and burnup.** Given the maximum allowable discharge burnup, is the current fuel geometry the best choice? Compare alternative rod arrangements, including square and triangular pitches, while distinguishing the rod lattice from the bundle envelope and the channel lattice in the moderator.

15. **Flow rate and coolant degradation.** Does faster circulation reduce degradation by shortening each pass through the core? Can reactions outside the irradiated region reverse any damage? Does the coolant approach a stable composition, or does degradation continue?

16. how to increase fuel utilization? decrase spread of BU between assemblies (assume similat to CANDU where we push the elements axially). figure of merit is average assembly BU, not max. since we might push k_effetive down but make it better to flatten the BU profile, we can have a vairalbe pitch of the chalandira tubes instead. this way as we get to the center the assemblies are further apart, we also need a better way to represent the lattice than what we have here, something that is amainable to an easier parametrixation than a triangular lattice. there are lots of ways to do this, different fuel enrichment etc. if we also push the fuel inside the channels as does CANDU then it depletes in that process and this should be taken into account. this means that we have more power generally speaking produced at the bottom half of the reactor.
17. I would separate the reactivity control mechanism from the chalandria and instead have dedicated channels that perform reactivity regulation (with boron solution)
18. we can perfrom axial and radial shuffling to increase max BU. our goal however is: lets settle a max BU limit, (25 MWd/kg), and see how we can meet it. we want all assemblies that are discharged to have the same max BU. the design space he is pretty big. we can remove assemblies in different rates from the core (hard limit we should have symmetry in the core, in the long term it will help more in optimizaiton and simplifications than constrain the design). DoFs: enrichment per ring, distance between rings, assemblies per ring, removal rate of assemblies per ring. How do feedbacks look like for all of those, are we under moderated? zoning is not a real issue here since we can make each channel have its own flow rate. how does excess reactivity look per iteration? also we want to look at the spread between each channel we want to make them as similar to each other as possible (similar to symmetry this will help down the line its easier to have many thermally and neutornically similar assemblies than a large spread).
19. are we considering about shuffling fuel from one end to another channel in the same end or all fuel addition is done in the same reactor side (top) and fuel removal from the same (bottom), we might have to fight coolant flow /gravity here so perhaps its better to only add fuelfrom top remove it from the bottom? this is a hard problem, and it definately needs a fast neutronics solver.
20. from first principles: the more excess reactivity, the more BU that can be produced.
21. its simpler to pass the fuel only once through the core so this should be the first approach. it also minimizes the fuel operations so its advantageous.
22.
17.main issues:
* coolant interaction with fuel (normal and off normal)
* coolant interaction with water (if a break in the chalandria or steam generator occurred). we could have an intermediate coolant IHX and then a steam generator to separate the organic primary coolant from the steam/water circuit, but this adds equipment and thermal resistance and does not eliminate the possibility of moderator ingress through a failed channel boundary.
* coolant properties, evolution

1. fuel management scheme (loading, shuffling, )
