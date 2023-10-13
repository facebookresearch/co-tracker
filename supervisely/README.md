## Supervisely Python SDK

This SDK aims to make it as easy as possible to develop new plugins for the
Supervisely platform. Check out also our guide on [how to package your plugin as
a Docker image](../help/tutorials/01_create_new_plugin/) and make it available
in the platform.

The SDK is a product of our experience developind Supervisely plugins and
contains functionality that we have found helpful and frequently needed for
pluginc development. The key areas covered by the SDK are
 * Reading, modifying and writing Supervisely projects on disk.
 * Working with labeling data: geometric objects and tags.
 * Common functionality for developing Supervisely plugins, such as neural
   network models.

The detailed documentation is still under active development. Here is a broad
outline of the available modules and areas they cover:
* `annnotation` - Working with labeling data of individual images. `Annotation`
  is the class that wraps all the labeling data for a given image: its `Label`s
  (geometrical objects)  and `Tag`s. See this tutorial [on how to work
  with annotations in detail](../help/jupyterlab_scripts/src/tutorials/01_project_structure/project.ipynb).
* `api` - Python wrappers to script your interactions with the Supervisely web
  instance. Instead of clicking around, you can write a script to request, via
  the API, a sequence of tasks, like training up a neural network and then
  running inference on a validation dataset.
* `aug` - Data augmentations to create more data variety for neural networks
  training.
* `geometry` - All the logic concerned with working with geometric objects - 
  compute statistics like object area, transform (rotate, scale, shift),
  extract bounding boxes, compute intersections and more.
* `imaging` - Our wrappers for working with images. IO, transformations,
  text rendering, color conversions.
* `io` - Low-level convenience IO wrappers that we found useful internally.
* `metric` - Computing similarity metrics between annotations. Precision,
  recall, intersection over union, mean average precision etc. Also includes
  wrappers to easily package metric computations as Supervisely plugins.
* `nn` - All the common code for working with neural networks. Includes a class
  to easily iterate over a Supervisely dataset during neural network training.
  Also conversions between raw tensors and Supervisely labeling data.
  * `hosted` - All the neural network logic that is specific to interacting with
    Supervisely platform. Base classes for neural network training and inference
    with common functionality already implemented (see our [guide on adding your
    own neural network plugin](../help/tutorials/03_custom_neural_net_plugin/custom_nn_plugin.md)),
    parsing, interpreting and consistency checking of configs, class indexing
    (between integer IDs of the tensors and Supervisely classes) helpers.
  * `pytorch` - Convenience wrappers for PyTorch - model weights IO, metrics
    etc.
  * `training` - A helper to flexibly schedule when to run validation during the
    model training.
* `project` - Working with Superrvisely projects on disk.
* `task` - Constants defining the directories where a plugin should expect the
  input and output data to be. Also helpers to stream progress data from a
  running plugin back to the web instance.
