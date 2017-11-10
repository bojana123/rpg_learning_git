import gflags

FLAGS = gflags.FLAGS


###################################
# DEFINE HERE FLAGS FOR YOUR MODEL#
###################################


# Train parameters
gflags.DEFINE_integer('img_width', 346, 'Target Image Width')
gflags.DEFINE_integer('img_height', 260, 'Target Image Height')
gflags.DEFINE_integer('crop_img_width', 200, 'Cropped image widht')
gflags.DEFINE_integer('crop_img_height', 200, 'Cropped image height')
gflags.DEFINE_integer('davis_img_width', 346, 'DAVIS Image Width')
gflags.DEFINE_integer('davis_img_height', 260, 'DAVIS Image Height')
gflags.DEFINE_integer('batch_size', 15, 'Batch size in training and evaluation')
gflags.DEFINE_float("learning_rate", 0.001, "Learning rate for adam optimizer")
gflags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
gflags.DEFINE_float("l2_reg_scale", 1e-4, "Scale for regularization losses")
gflags.DEFINE_integer('output_dim', 1, "Number of outputs")
gflags.DEFINE_integer("max_epochs", 100, "Maximum number of training epochs")
gflags.DEFINE_string('frame_mode', "dvs", 'Load mode for images, either '
                     'dvs, aps or aps_diff')


###############################################################
# MAKE SURE TO CONFIG THIS PARAMETER SUCH THAT YOUR GPU USAGE #
# (that you can see with `$ nvidia-smi`) IS AT LEAST 60%       #
###############################################################
gflags.DEFINE_integer('num_threads', 8, 'Number of threads reading and '
                      '(optionally) preprocessing input files into queues')
gflags.DEFINE_integer('capacity_queue', 100, 'Capacity of input queue. A high '
                      'number speeds up computation but requires more RAM')

# Reading parameters
gflags.DEFINE_string('train_dir', "../training", 'Folder containing'
                     ' training experiments')
gflags.DEFINE_string('checkpoint_dir', "./tests/test_0/", "Directory name to"
                     "save checkpoints and logs.")

# Log parameters
gflags.DEFINE_integer("max_steps", 200000, "Maximum number of training iterations")
gflags.DEFINE_bool('resume_train', False, 'Whether to restore a trained'
                   ' model for training')
gflags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations")
gflags.DEFINE_integer("save_latest_freq", 5000, \
    "Save the latest model every save_latest_freq iterations"
                     "(overwrites the previous latest model)")

# Testing parameters
gflags.DEFINE_string('test_dir', "../testing", 'Folder containing'
                     ' testing experiments')
gflags.DEFINE_string('output_dir', "./tests/test_0", 'Folder containing'
                     ' testing experiments')
gflags.DEFINE_string("ckpt_file", None, "Checkpoint file")
