package org.atlanmod.slepaper.modeling.generators;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;
import java.util.Map;
import java.util.Properties;

import org.atlanmod.emfviews.core.Viewpoint;
import org.atlanmod.emfviews.core.ViewpointResource;
import org.atlanmod.emfviews.virtuallinks.ConcreteConcept;
import org.atlanmod.emfviews.virtuallinks.VirtualLinksFactory;
import org.atlanmod.emfviews.virtuallinks.VirtualLinksPackage;
import org.atlanmod.erpaper.modeling.utils.EmfViewsFactory;
import org.atlanmod.erpaper.modeling.utils.ViewUtils;
import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EAttribute;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.EcoreResourceFactoryImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.stream.JsonReader;

public class GenerateWeavingModel {
	public static String here = new File(".").getAbsolutePath();

	public static URI resourceURI(String relativePath) {
		return URI.createFileURI(here + relativePath);
	}

	public static void main(String[] args) throws Exception {

		//Paths for Example View files
		String predictedViewDirectory = "/../Views/Recommended_View/";
		String eViewFile = predictedViewDirectory + "my_view/recommended.eview";
		String eViewPointFile = predictedViewDirectory + "src-gen/recommended.eviewpoint";
		String viewpointWMFile = predictedViewDirectory + "src-gen/recommended.xmi";
		String jsonPredictedFile = predictedViewDirectory + "for_recommendation/recommendations.json";
		
		//Paths for modeling Resources
		String modelingResourcesDirectory = "/../Modeling_Resources/metamodels/";
		
		// Create basic resources to deal with EMF reflective API
		Map<String, Object> map = Resource.Factory.Registry.INSTANCE.getExtensionToFactoryMap();
		map.put("xmi", new XMIResourceFactoryImpl());
		map.put("*", new EcoreResourceFactoryImpl());
		map.put("eviewpoint", new EmfViewsFactory());
		map.put("eview", new EmfViewsFactory());
		
		
		// Make sure the weaving model package is loaded
		VirtualLinksPackage.eINSTANCE.eClass();

		//global ResourceSet and baseRegistry
		ResourceSet rSet = new ResourceSetImpl();
		
		// Register metamodels
		//TODO: How to get from global register?
		EPackage aPkg = (EPackage) rSet.getResource(resourceURI(modelingResourcesDirectory + "A.ecore"), true).getContents().get(0);
		EPackage.Registry.INSTANCE.put(aPkg.getNsURI(), aPkg);
		EPackage bPkg = (EPackage) rSet.getResource(resourceURI(modelingResourcesDirectory + "B.ecore"), true).getContents().get(0);
		EPackage.Registry.INSTANCE.put(bPkg.getNsURI(), bPkg);
		
		//Load Viewpoint and Viewpoint WeavingModel
		ViewpointResource vResource = new ViewpointResource(resourceURI(eViewPointFile));
		vResource.load(null);
		Viewpoint viewpoint = vResource.getViewpoint();
		
		Resource viewpointWM = rSet.getResource(resourceURI(viewpointWMFile), true);
		viewpointWM.load(null);
		
		//Get Contribution Packages
		Map<String, EPackage> viewPointContribEPackages = viewpoint.getContributingEPackages();
		
		//Read .eview as properties since there is no weaving model
		Properties propsEView = new Properties();
		propsEView.load(new FileReader(here + eViewFile));
		String contributingModels = propsEView.getProperty("contributingModels");
		
		String[] viewContribModelsString = contributingModels.split(",");
		String[] aliasPathPairLeft = viewContribModelsString[0].split("::");
		String classLeft = aliasPathPairLeft[0];
		String modelLeft = aliasPathPairLeft[1].replace("../../", "/../");
		
		String[] aliasPathPairRight = viewContribModelsString[1].split("::");
		String classRight = aliasPathPairRight[0];
		String modelRight = aliasPathPairRight[1].replace("../../", "/../");

		// Create a new instance of Utils class to work with View elements
		// including a factory for the Virtual Links
		VirtualLinksFactory vLinksFactory = VirtualLinksFactory.eINSTANCE;
		ViewUtils viewUtils = new ViewUtils(vLinksFactory);

		// Build view weaving model
		viewUtils.createWeavingModel("recommended", true);//TODO: get whitelist from viewpoint
		
		//Set Contributing models in ViewLevel
	    viewUtils.createContributingModels(viewPointContribEPackages);
	    
		
		try {
		    
			JsonReader readerJson = new JsonReader(new FileReader(here + jsonPredictedFile));
			JsonElement jsonElement = JsonParser.parseReader(readerJson);
		    JsonObject jsonObject = jsonElement.getAsJsonObject();

		    for (Map.Entry<String, JsonElement> entry : jsonObject.entrySet()) {
		        String key = entry.getKey();
		        JsonElement value = entry.getValue();
		        
		        //create the concrete element class left
		        EPackage contribModelLeft = viewPointContribEPackages.get(classLeft);
		        String pathLeft = getPathFromModel(contribModelLeft.getNsURI(), modelLeft, classLeft, key);
		        ConcreteConcept leftConcept = viewUtils.createConcreteConcept(contribModelLeft.getNsURI(), pathLeft);
		        		        
		        JsonArray jsonArray = value.getAsJsonArray();

		        int count = 1;
		        for (JsonElement elementRightID : jsonArray) {
		        	//create the concrete element class right
			        EPackage contribModelRight = viewPointContribEPackages.get(classRight);
			        String pathRight = getPathFromModel(contribModelRight.getNsURI(), modelRight, classRight, elementRightID.toString());
			        ConcreteConcept rightConcept = viewUtils.createConcreteConcept(contribModelRight.getNsURI(), pathRight);
			        
			        viewUtils.createVirtualAssociation("relates_with", leftConcept, rightConcept, -1);
			        count++;
			        
			        if (count == 3) {
			        	break;
			        }
		        }
		    }

		} catch (Exception ex) {
		    ex.printStackTrace();
		}		
		
		EObject viewWM = (EObject) viewUtils.getWeavingModel();
		Resource serializedViewWM = null;
		URI uriSerializedViewWM;
		uriSerializedViewWM = resourceURI(predictedViewDirectory + "my_view/recommended.xmi");
		serializedViewWM = rSet.createResource(uriSerializedViewWM);
		serializedViewWM.getContents().add(viewWM);
		// serialize
		try {
			serializedViewWM.save(null);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static String getPathFromModel(String nsURi, String modelLeft, String classLeft, String valueToSearch) throws IOException {
		ResourceSet resourceSet = new ResourceSetImpl();
		Resource myModel = resourceSet.getResource(resourceURI(modelLeft), true);

		myModel.load(null);
		
		for (Iterator<EObject> i = (Iterator<EObject>) myModel.getAllContents(); i.hasNext();) {
		    EObject object = i.next();

		    // Check if the object has the attribute
		    //TODO: Replace ID by the attribute selected by the user
		    EAttribute attribute = (EAttribute) object.eClass().getEStructuralFeature("ID");
		    if (attribute != null && object.eGet(attribute).toString().equals(valueToSearch)) {
		        // found the object
		    	return myModel.getURIFragment(object);
		    }
		}
		
		return null;
	}
}
