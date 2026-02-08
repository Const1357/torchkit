# TODO: design abstractions and implement.

# ideas: 
# 1. SINGLE backbone + MULTIPLE heads (e.g. for classification and regression at the same time)
# 2. there has to be a way to route which features go to which head during forward pass
#    - maybe the backbone can return a dict of features and during forward they will be sent to the appropriate head based on some key?
#    - the keys are user-defined and instantiated in the constructor (kwargs in ModuleFactory)