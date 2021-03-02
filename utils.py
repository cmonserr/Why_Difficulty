import tensorflow as tf

def clean_list_of_models_names(lists_of_private_object):
  #todos los modelos empiezan por mayuscula y no son privados, es decir no pueden empezar con "_"
  prev_clean_list=list()
  clean_list=list()
  for element in lists_of_private_object:
    if element[0]=="_" or not element[0].isupper():
      pass
    else:
      
      prev_clean_list.append(element)
  return prev_clean_list

def clean_list_of_familys(lists_of_private_object):
  #todos los modelos empiezan por mayuscula y no son privados, es decir no pueden empezar con "_"
  prev_clean_list=list()
  clean_list=list()
  for element in lists_of_private_object:
    if element[0]=="_" or not element[0].islower() or element=="imagenet_utils":
      pass
    else:      
      prev_clean_list.append(element)
  return prev_clean_list
  
def get_preprocesing_input(ALL_FAMILYS,model):
  def check_which_is_family_is_the_family_to_which_it_belongs(ALL_FAMILYS,model):

    for family in ALL_FAMILYS:
      family_module=getattr(tf.keras.applications,family)
      if model in dir(family_module):
        return family_module
    return "ERROR, NO SE HA ENCONTRADO EN NINGUNA FAMILIA"
  def get_function_preprocess_input(module_family_net):
    return getattr(module_family_net,"preprocess_input")
  family_module=check_which_is_family_is_the_family_to_which_it_belongs(ALL_FAMILYS,model)
  preprocess_input=(get_function_preprocess_input(family_module))
  return preprocess_input