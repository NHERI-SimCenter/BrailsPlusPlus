Hazus earthquake inventory tables
=================================

``HazusEQStructureTypeInferer``, ``HazusEQDesignLevelInferer`` and
``HazusEQHeightClassInferer`` draw from tables of the FEMA Hazus Inventory
Technical Manual, transcribed in ``hazus_rulesets.py``, each by a getter
whose docstring names the manual table it transcribes. This page states
which editions of the manual the tables come from, what the completion of
the West Coast tables in ``hazus_table_completion.py`` rests on, and the
scope of the rule on unreinforced structure types.

The transcribed tables
----------------------

The Appendix A structure type weights, which ``get_hazus_occ_type_mapping``
returns, come from the 6.0 edition of the manual:
https://www.fema.gov/sites/default/files/documents/fema_hazus-6-inventory-technical-manual.pdf.
The 6.1 edition (August 2024) renumbered that appendix and replaced the
state-level weight tables with distributions conditioned on the general
building type, so the weights in ``hazus_rulesets.py`` appear in print only
in 6.0. The replacement cost, garage and income tables
(``get_hazus_base_replacement_cost``, ``get_hazus_region_to_garage`` and
``get_hazus_income_to_const_class``) follow the 6.1 edition and cite its
numbering.

The design levels that admit URM, C3 and S5
(``get_hazus_unreinforced_design_levels``) transcribe a footnote under the
Appendix A structure type tables of the 7.0 edition. The 6.1 edition
carries the same footnote, and the 6.0 edition carries none. Appendix A of
6.1 and 7.0 is cited by page: both editions print two tables captioned
A-1, so every caption after the first sits one below the number the
manual's own cross-references use.

The completed West Coast tables
-------------------------------

The West Coast mid-rise and high-rise tables of the 6.0 edition, Appendix A
(Tables A-5 to A-10, pp. A-5 to A-8), print no distribution for some cells
of occupancy class, story band and era: 71 in the transcription that
``get_hazus_occ_type_mapping`` returns. A search of the 6.0, 6.1 and 7.0
editions found no published row for any of them. The 6.0 edition omits
some of these rows and prints others with a label and no values, and 6.1
and 7.0 omit every one. ``hazus_table_completion.py`` fills each cell from
printed rows of the transcription. The completion is this project's own,
and no edition of the manual prints it.

A cell takes the first of two sources that keeps at least ``HALF_MASS``
of its weight on the type list of the cell's story band once URM, C3 and
S5 are set aside: the row of the same occupancy class and story band in a
neighboring era, then the row of the same occupancy class and era one
story band down. The three types are set aside in that test because the
region bars them above its earliest design level, and a building of any
era may be given a later one. A cell that neither source fills combines
the printed rows of its story band and era within each general building
type. Cells of the pairs in ``IMPLAUSIBLE_PAIRS`` are filled too, and
never serve as sources. ``get_west_coast_fill_routes`` states the rule in
full.

Scope
-----

The footnote bars URM, C3 and S5 above Pre-Code in the West Coast region
group and above Low-Code in the other two. Its condition is the region
group, which selects a table set and spans a wide range of seismicity: the
West Coast group runs from UBC zone 4 on the California coast to zones 0
and 1 across the interior. The stock it governs is nearly empty. In the
West Coast tables, the three types carry 28.18% of the printed weight of the
``pre_1950`` era, 1.90% of the ``1950_to_1970`` era and 0.08% of the
``post_1970`` era, while every Low-Code year in the generic design level
table falls after 1976.

The footnote is the tighter of the two rules that bear on these types. The
Hazus Earthquake Model Technical Manual 6.1, Table 5-7 (p. 5-27),
parameterizes them under Pre-Code and Low-Code only, nationwide, and the
fragility library ships that bound. A caller who would apply it on the West
Coast passes ``unreinforced_design_levels=('Pre-Code', 'Low-Code')`` to
``HazusEQStructureTypeInferer``.
