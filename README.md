caterpillar-lsi
===============
Latent Semantic Indexing plugin for Caterpillar.

Usage
-----
You can create a Latent Semantic model for an index by running the plugin:

    with IndexWriter(path) as writer:
        writer.run_plugin(LSIPlugin, num_features=300, normalise_frequencies=True)
    
...and then generate a list of similarities between all frames in an index and all frames in the model:

    with IndexReader(path) as reader:
        similarities = LSIPlugin(reader).compare_index(reader)

  
Credits
-------
This plugin was initally developed by [Kris Rogers](https://github.com/krisrogers) and later updated by [Ryan Stuart](https://github.com/rstuart85). This work was supported by the University of Queensland Vice Chancellor’s Strategic Initiative Grant in Language Technologies.

License
=======
caterpillar-lsi is copyright Kapiche Limited. It is licensed under the GNU Affero General Public License.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

The copyright holders grant you an additional permission under Section 7 of the GNU Affero General Public License, version 3, exempting you from the requirement in Section 6 of the GNU General Public License, version 3, to accompany Corresponding Source with Installation Information for the Program or any work based on the Program. You are still required to comply with all other Section 6 requirements to provide Corresponding Source.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
