# Paper Title

Repository containing all artifacts for the paper "Title" submitted to " ". The repository includes all necessary files to execute the tool for recommended Model Views and the code for evaluation.

## Getting started

The final part of this README file contains a tree view of the repo organization. Each main directory includes a detailed extra README inside it with detailed info. Below is the basic flow to use the tool and run the evaluation steps.

## Use the tool

The tool aims to automatically recommend links in Model Views by training a GNN architecture with previously known links between Heterogeneous EMF models.

### Requirements

- It is necessary to have a working instance of Eclipse Modeling Workbench with the EMF Views plugin installed.
- Python >= 3.8 (refer to the ML directory for detailed instructions)

### How to generate the necessary files for execution

1. Using the Eclipse instance, run the Java Program `GenerateExample` in the `org.atlanmod.erpaper.modeling` project to get the example files used in the steps decribed below
    - Into `Modeling_Resources/models`, the program will create 2 .xmi files with the serialized models and a CSV with the defined relations.
 
### How to execute step-by-step

1. Create the VPDL file defining the metamodels, but without the where clause
    - Important to point out the classes that participate in the relation and the relation name
    - This will generate the .eviewpoint file and the viewpoint WeavingModel (.xmi)
    - E.g. The `src` and `src-gen` directories into `Recommended_View` project.
2. Create the actual view folder with the standard .eview file
    - Point to the generated eviewpoint file, the contributing models, and to a not existing yet View weaving model
    - Create the extra  parameters `.gnn` file with important information for the Machine Learning execution
    - E.g. The `.eview` file into the `my_view` directory into `Recommended_View` project.
3. Optionally inthe parameters file include the unique ID of the models and attributes to be ignored in the learning process.
    - If ID is not provided, the tool will consider the URI fragment of the element
    - If ignored attributes are not provided, the tool will encode everything
4. Run the `GenerateMLData` java program in the `org.atlanmod.erpaper.modeling`
    - This will generate the filtered models (as .xmi serializations) and a .json parameters file to be used in Machine Learning into the `Data` directory.
5. Run the tool using the provided Notebook/Script in the ML folder following the notebook cells instructions
6. Run the `GenerateWeavingModel` java program in the `org.atlanmod.erpaper.modeling` to get the view WeavingModel with the recommended links
7. Open the previous mentioned `.eview` file with a model browser (e.g. MoDisco) and check the view with the recommended links

### Folder structure
