<p align="center">
 <b>BRAILS++: Building Regional Asset Inventories at Large Scale</p>
</p>

## What is it?

'Brails++' is an object-oriented framework for building applications that focus on generate asset inventories for large geographic regions.


UNDER CONSTRUCTION!!


## How is the Repo laid out

<ul>
<li>brails: A directory containing the classes
<ul>
  <li>brails/types: directory containing useful datatypes, e.g. ImageSet and AssetInventory</li>
  <li>brails/processors: directory containing classes that do image_processing to make predictions, e.g. RoofShape</li>
  <li>brails/segmenters: directory containingg classes that do image segmentation</li>
  <li>brails/scrapers: directory containing classes that do internet downloads, e.g. footprint scrapers, image scrapers</li>
  <li>brails/filters: directory containiig image filters, e.g. classes that take images and revise or throw out</li>
  <li>brails/imputaters: directory containing classes that fill in missing AssetInventory datasets</li>
  <li>brails/utils: directory containing misc classes that do usefule stuff, e.g. geometric conversions</li>
</ul>
</li>
<li>examples: A directory containing examples
</li>
</ul>