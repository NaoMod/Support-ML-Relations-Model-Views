from pyecore.resources import ResourceSet, URI
import glob
from os import listdir, path as osp
from pathlib import Path

class Metamodels():
    
    def __init__(self):
        """Initialize the Metamodels class.

        This method initializes the Metamodels class by creating a ResourceSet
        and setting the modeling_resources_path attribute.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.resource_set = ResourceSet()
        self.modeling_resources_path = glob.glob(osp.join(Path(__file__).parent, '..','..','Modeling_Resources'))[0]

    def register(self):
        """Register the metamodels in the resource set.

        Returns
        -------
        None

        Raises
        ------
        FileNotFoundError
            If the metamodels directory is not found.
        """
        directory = osp.join(self.modeling_resources_path, 'metamodels')
        files = listdir(directory)
        ecore_paths = [f for f in files if osp.isfile(osp.join(directory,f))]
        
        for ecore_file in ecore_paths:

            ecore_path = osp.join(directory,ecore_file)
            resource_path = self.resource_set.get_resource(URI(ecore_path))
            root_pkg = resource_path.contents[0]
            
            if hasattr(root_pkg, 'nsURI') and root_pkg.nsURI != "":
                self.resource_set.metamodel_registry[root_pkg.nsURI] = root_pkg
            
            contents = root_pkg.eContents

            for content in contents:
                if hasattr(content, 'nsURI') and content.nsURI != "":
                    self.resource_set.metamodel_registry[content.nsURI] = content

    def get_resource_set(self):
        """Return the resource set.

        Returns
        -------
        ResourceSet
            The resource set object.
        """
        return self.resource_set
