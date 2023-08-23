import os
from pyecore.resources import ResourceSet, URI
from pyecore.resources.json import JsonResource

# Set the directory for the Ecore and JSON files
here = os.path.abspath(".")

def resource_uri(relative_path):
    return URI(os.path.join(here, relative_path))

def load_ecore_models():
    resource_set = ResourceSet()

    
    # Load the Viewpoint and Viewpoint WeavingModel
    viewpoint_resource = resource_set.get_resource(resource_uri("../Views/Recommended_View/src-gen/recommended.eviewpoint"))
    viewpoint_resource.load()

    # Load the JSON file
    with open(os.path.join(here, "../Views/Recommended_View/for_recommendation/recommendations.json")) as json_file:
        json_data = json_file.read()

    # Parse the JSON data
    json_resource = JsonResource()
    json_resource.load_str(json_data)

    # Get the Viewpoint and Viewpoint WeavingModel instances
    viewpoint = viewpoint_resource.contents[0]
    viewpoint_wm = resource_set.get_resource(resource_uri("../Views/Recommended_View/src-gen/recommended.xmi")).contents[0]

    # Get Contribution Packages
    view_point_contrib_e_packages = viewpoint.get_contributing_epackages()

    return viewpoint_wm, view_point_contrib_e_packages, json_resource

def create_concrete_concept(viewpoint_contrib_epackages, ns_uri, path):
    # Get the EPackage based on the nsURI
    e_pkg = viewpoint_contrib_epackages[ns_uri]

    # Get the EClass based on the path
    e_class = e_pkg.get_e_classifier(path)

    # Create and return a new instance of the EClass
    return e_class()

def main():
    viewpoint_wm, view_point_contrib_e_packages, json_resource = load_ecore_models()

    # Create a new instance of Utils class to work with View elements
    # including a factory for the Virtual Links
    virtual_links_factory = VirtualLinksFactory()

    # Create a new ViewUtils instance
    view_utils = ViewUtils(virtual_links_factory)

    # Build view weaving model
    view_utils.create_weaving_model("recommended", True)  # TODO: get whitelist from viewpoint

    # Set Contributing models in ViewLevel
    view_utils.create_contributing_models(view_point_contrib_e_packages)

    for key, value in json_resource:
        # Create the concrete element class left
        class_left, model_left = value[0].split("::")
        path_left = get_path_from_model(view_point_contrib_e_packages[class_left].nsURI, model_left, class_left, key)
        left_concept = create_concrete_concept(view_point_contrib_e_packages, view_point_contrib_e_packages[class_left].nsURI, path_left)

        count = 1
        for element_right_id in value[1]:
            # Create the concrete element class right
            class_right, model_right = value[1][element_right_id].split("::")
            path_right = get_path_from_model(view_point_contrib_e_packages[class_right].nsURI, model_right, class_right, element_right_id)
            right_concept = create_concrete_concept(view_point_contrib_e_packages, view_point_contrib_e_packages[class_right].nsURI, path_right)

            view_utils.create_virtual_association("relates_with", left_concept, right_concept, -1)

            count += 1
            if count == 3:
                break

    # Serialize the Viewpoint WeavingModel
    serialized_view_wm = resource_set.create_resource(resource_uri("../Views/Recommended_View/my_view/recommended.xmi"))
    serialized_view_wm.append(viewpoint_wm)
    serialized_view_wm.save()
