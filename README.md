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
2. Create the actual view folder with the standard .eview file
    - Point to the generated eviewpoint file, to the models, and to a not existing yet View weaving model
    - Create the extra parameters.gnn file with some important information for the Machine Learning execution
3. Optionally include the unique ID of the models and attributes to be ignored in the learning process.
    - If ID is not provided, the tool will consider the URI fragment of the element
    - If ignored attributes are not provided, the tool will encode everything
4. Run the tool using the provided Notebook/Script in the ML folder
5. Run the `GenerateWeavingModel` java program in the `org.atlanmod.erpaper.modeling` to get the view WeavingModel with the recommended links

### Folder structure
