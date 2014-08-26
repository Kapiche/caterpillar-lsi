caterpillar-lsi
===============
Latent Semantic Indexing plugin for Caterpillar.

**Warning** - This is a prototype implementation that is currently only compatible with Caterpillar commit a984bfd34fe666d2d77c2f8917ca580946cb509f 

Usage
=====
You can create a Latent Semantic model for an index by running the plugin:

    lsi_plugin = index.run_plugin(LSIPlugin, num_features=300, normalise_frequencies=True)
    
...and then generate a list of similarities between a specified frame and all frames in the model:

    similarities = lsi_plugin.compare_document(index, QueryStringQuery("id=frame-1"))

  
License
=======
[Apache 2.0 License](http://www.apache.org/licenses/LICENSE-2.0.html)
