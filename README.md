## Integrating the Support for Machine Learning of Inter-Model Relations in Model Views

**PRIVATE REPOSITORY. AVAILABLE JUST ANONYMOUSLY**

This repository contains all the artifacts for the paper "Integrating the Support for Machine Learning of Inter-Model Relations in Model Views, " submitted to the 20th European Conference on Modelling Foundations and Applications (ECMFA 2024).

The repository is the artifact companion for the paper and includes all necessary instructions to execute the tool.

### Environment preparation ⚙️

How to prepare the environment to execute the tool.

#### Requirements

- It is necessary to have a working instance of Eclipse Modeling Workbench with the [EMF Views](https://github.com/atlanmod/emfviews/tree/AIDOaRt_new_vpdl) plugin installed (branch `AIDOaRt_new_vpdl`).
    - TODO: Include a link for an updated site with the correct branch for EMF Views
- Python >= 3.8 (refer to the `Python` directory for detailed instructions on how to set up the virtual environment)

#### Projects

- The directories `Java`, `Modeling_Resources`, and `Views` are all Eclipse-configured projects that can be imported to a running Eclipse instance. They include 9 Eclipse projects.
- The `Python` directory includes the Python files necessary to execute the "View Learning" component of the solution. We recommend running them using a VS Code instance.

### Use the tool 🚀

The tool aims to automatically recommend links in Model Views by training a GNN architecture with previously known links between Heterogeneous EMF models.

#### Generating the files for the tests

1. Into the Java project `org.atlanmod.slepaper.modeling.generators`, run the Java program `UserMoviesExample` to generate the example for the Users/Movies use case described in section 2.3 of the paper.
    - This will download the data from [MovieLens](https://grouplens.org/datasets/movielens/) dataset and generate the models into `Modeling_Resources/models`.
    - To check the models in the browser, it may be necessary to register the metamodels into the package registry. To do so, check the Ecore metamodels into `Modeling_Resources/metamodels`.
2. Into the Java project `org.atlanmod.slepaper.modeling.generators`, run the Java program `ABExample` to generate the example for the AB use case used in the evaluation section of the paper (Section 5).

All example files will be stored in the `Modeling_Resources` project under the `models/Generated` directory.

With these generated files, it's possible to use the tool to replicate the paper's results as to use recommendations for any other view.
 
#### Executing the examples step-by-step

All example views are stored as Xtext projects into the directory called `Views`.

*Executing the Users/Movies example*

1. Open the VPDL file defining the metamodels, but without the where clause
    - It is essential to point out the classes that participate in the relation and the relation name
    - This will generate the .eviewpoint file and the viewpoint WeavingModel (.xmi)
    - E.g., The `src` and `src-gen` directories into `Recommended_View` project.
2. Create the actual view folder with the standard .eview file
    - Point to the generated eviewpoint file, the contributing models, and an existing yet View weaving model
    - Create the extra  parameters `.gnn` file with important information for the Machine Learning execution
    - E.g., The `.eview` file into the `my_view` directory into `Recommended_View` project.
3. Optionally, the parameters file includes the unique ID of the models and attributes to be ignored in the learning process.
    - If ID is not provided, the tool will consider the URI fragment of the element
    - If ignored attributes are not provided, the tool will encode everything
4. Run the `GenerateMLData` java program in the `org.atlanmod.erpaper.modeling`
    - This will generate the filtered models (as .xmi serializations) and a .json parameters file for Machine Learning into the `Data` directory.
5. Run the tool using the provided Notebook/Script in the ML folder following the notebook cells' instructions
6. Run the `GenerateWeavingModel` java program in the `org.atlanmod.erpaper.modeling` to get the view WeavingModel with the recommended links
7. Open the previously mentioned `.eview` file with a model browser (e.g., MoDisco) and check the view with the recommended links

### Folder structure
