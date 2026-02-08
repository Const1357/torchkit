# write the class for an MLP classifier head.

# ensure that its architecture is fully parametrizable for maximum flexibility.

# it should take the classes as an argument, and use that to determine the output dimension.
# it should also take the depth and widths of the MLP as arguments.

# it should expose options for batchnorm/layernorm, dropout, and activation functions.
# it should be compatible with the SKTorchClassifier interface, and be able to be instantiated by a ModuleFactory.
