## Integrating the Support for Machine Learning of Inter-Model Relations in Model Views

This repository contains all the artifacts for the paper "Integrating the Support for Machine Learning of Inter-Model Relations in Model Views, " submitted to the 20th European Conference on Modelling Foundations and Applications (ECMFA 2024).

The repository is the artifact companion for the paper and includes all necessary instructions to execute the tool and reproduce the results.

### Environment preparation ⚙️

How to prepare the environment to execute the tool.

#### Requirements

- It is necessary to have a working instance of Eclipse Modeling Workbench with the [EMF Views](https://github.com/atlanmod/emfviews/tree/AIDOaRt_new_vpdl) plugin installed (using the branch `AIDOaRt_new_vpdl`).
- Python >= 3.8 (refer to the `Python` directory for detailed instructions on how to set up the virtual environment)

#### Projects

- The directories `Java`, `Modeling_Resources`, and `Views` are all Eclipse-configured projects that can be imported to a running Eclipse instance. They include 9 Eclipse projects.
- The `Python` directory includes the Python files necessary to execute the "View Learning" component of the solution. We recommend running them using a VS Code instance.

### Use the tool to reproduce the paper results 🚀

The tool aims to automatically recommend links in Model Views by training a GNN architecture with previously known links between Heterogeneous EMF models.

#### How to reproduce

The directory `Modeling_Resources` contains all necessary models to reproduce the experiments and the Python scripts already use this directory to execute. 
After following the instruction into `Python` directory to install dependencies, the following commands can be executed to reproduce the results.

##### Movies_Users example

1. `python main.py Movies_Users`
2. The script will get all modeling resources, train the GNN and get the inferences.
3. The Weaving model will be stored in the respective Movies_Users View project, It can be opened into the Eclipse enviroment.
4. The script will also save a PDF file at root level with the generated ROC curve.

##### AB example

The process is the same as above, but using the following parameters instead of Movies_Users:

- AB1
- AB2
- AB3

The generated Weaving models and curves has the name of the parameter

### Extra

#### Regerate the model examples

If you want to regenarate the models used for the experiments, execute the following:

1. Into the Java project `org.atlanmod.gnnpaper.modeling.generators`, run the Java program `UserMoviesExample` to generate the example for the Movies_Users use case described.
    - This will download the data from [MovieLens](https://grouplens.org/datasets/movielens/) dataset and generate the models into `Modeling_Resources/models`.
    - To check the models in the browser, it may be necessary to register the metamodels into the package registry. To do so, check the Ecore metamodels into `Modeling_Resources/metamodels`.
2. Into the Java project `org.atlanmod.gnnpaper.modeling.generators`, run the Java program `ABExample` to generate the examplse for the AB use case used in the evaluation section of the paper.

All example files will be stored in the `Modeling_Resources` project under the `models/Generated` directory.

Note: The new generation can affect the results presented in the paper.
 
#### Using the updated VPDL

All example views are stored as Xtext projects into the directory called `Views`.

1. Open the VPDL file defining the metamodels, but without the where clause
    - It is essential to point out the classes that participate in the relation and the relation name
    - This will generate the .eviewpoint file and the viewpoint WeavingModel (.xmi)
    - E.g., The `src` and `src-gen` directories into `Recommended_View` project.
2. Create the actual view folder with the standard .eview file
    - Point to the generated eviewpoint file, the contributing models, and an existing yet View weaving model
    - Create the extra  parameters `.gnn` file with important information for the Machine Learning execution
3. Open the previously mentioned `.eview` file with a model browser (e.g., MoDisco) and check the view with the recommended links
