package org.atlanmod.slepaper.modeling.generators;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import org.atlanmod.emfviews.core.Viewpoint;
import org.atlanmod.emfviews.core.ViewpointResource;
import org.atlanmod.emfviews.virtuallinks.VirtualLinksPackage;
import org.atlanmod.erpaper.modeling.utils.EmfViewsFactory;
import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EAttribute;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EStructuralFeature;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.EcoreResourceFactoryImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;

import com.google.gson.Gson;

public class GenerateMLData {

	public static String here = new File(".").getAbsolutePath();

	public static URI resourceURI(String relativePath) {
		return URI.createFileURI(here + relativePath);
	}

	/**
	 * 
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		
		String DIRECTORY = "AB";		
		
		//Paths for Example View files
		String recommendedViewDirectory = "/../Views/Recommended_View/";
		
		String parametersFile = recommendedViewDirectory + "my_view/recommended.gnn";
		String eViewFile = recommendedViewDirectory + "my_view/recommended.eview";
		String eViewPointFile = recommendedViewDirectory + "src-gen/recommended.eviewpoint";
		String viewpointWMFile = recommendedViewDirectory + "src-gen/recommended.xmi";
		
		//Paths for modeling Resources with the metamodels
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
		String modelTrainingLeft = aliasPathPairLeft[1].replace("../../", "/../");
		
		String[] aliasPathPairRight = viewContribModelsString[1].split("::");
		String classRight = aliasPathPairRight[0];
		String modelTrainingRight = aliasPathPairRight[1].replace("../../", "/../");
		
		Properties props = new Properties();
		props.load(new FileReader(here + parametersFile));
		
		URI uriLeft = resourceURI("/../Data/" + DIRECTORY + "/DatasetLeft.xmi");
		Resource modelLeft = rSet.createResource(uriLeft);
		EPackage ePackageLeft = EPackage.Registry.INSTANCE.getEPackage(viewPointContribEPackages.get(classLeft).getNsURI());
		URI uriRight = resourceURI("/../Data/" + DIRECTORY + "/DatasetRight.xmi");
		EPackage ePackageRight = EPackage.Registry.INSTANCE.getEPackage(viewPointContribEPackages.get(classRight).getNsURI());
		Resource modelRight = rSet.createResource(uriRight);

		// TODO: Encapsulate in a for loop when have more models
		URI uriTrainingLeft = resourceURI(modelTrainingLeft);
		Resource modelForTrainingLeft = rSet.getResource(uriTrainingLeft, true);
		URI uriTrainingRight = resourceURI(modelTrainingRight);
		Resource modelForTrainingRight = rSet.getResource(uriTrainingRight, true);

		copyModel(modelForTrainingLeft, modelLeft, ePackageLeft, classLeft);
		copyModel(modelForTrainingRight, modelRight, ePackageRight, classRight);
		
		String csvPath = here + "/../Data/" + DIRECTORY + "/Relations.csv";
		
		createCSVSkeleton(csvPath, "Left_id", "Right_id");
		
//		String jsonPath = here + "/../Data/" + DIRECTORY + "/Parameters.json";
//		
//		props.put("CLASS_LEFT", classLeft);
//		props.put("CLASS_RIGHT", classRight);
//		createJsonParams(jsonPath, props);

		// serialize XMI files
		try {
			modelLeft.save(null);
			modelRight.save(null);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Create the skeleton of the CSV file with mandatory fields
	 * 
	 * @param csvPath     Path for the generated file
	 * @param LeftIdName  Name of the identifier of left model in the relation (source)
	 * @param RightIdName Name of the identifier of right model in the relation (target)
	 * 
	 * @throws IOException
	 */
	private static void createCSVSkeleton(String csvPath, String LeftIdName, String RightIdName) throws IOException {
		boolean fileExists = new File(csvPath).exists();
        FileWriter csvWriter = new FileWriter(csvPath, true);
        if (!fileExists) {
            csvWriter.append(LeftIdName);
            csvWriter.append(",");
            csvWriter.append(RightIdName);
            csvWriter.append("\n");
        }
        
        csvWriter.flush();
        csvWriter.close();
	}

	/**
	 * 
	 * @param modelFrom
	 * @param modelTo
	 * @param ePackageTo
	 * @param className
	 * @return
	 */
	public static Resource copyModel(Resource modelFrom, Resource modelTo, EPackage ePackageTo, String className) {
		List<EObject> elementsFrom = modelFrom.getContents();
		
		for (Iterator<EObject> iter = elementsFrom.iterator(); iter.hasNext();) {
			EObject elementFrom = iter.next();
			EClass modelClassFrom = elementFrom.eClass();
			String classNameFrom = modelClassFrom.getName();
			
			EObject objectModelTo = null;
			if (classNameFrom.equals(className)) {
				// include in the modelTo
				EClass eClassTo = (EClass) ePackageTo.getEClassifier(className);
				objectModelTo = ePackageTo.getEFactoryInstance().create(eClassTo);
				
				for (Iterator<EAttribute> iterAttr = modelClassFrom.getEAllAttributes().iterator(); iterAttr.hasNext();) {
					EAttribute elementAttribute = (EAttribute) iterAttr.next();

					Object elementAttributeValue = elementFrom.eGet(elementAttribute);
					String attrName = elementAttribute.getName();

					EStructuralFeature modelToFeature = objectModelTo.eClass().getEStructuralFeature(attrName);
					if (modelToFeature != null) {
						objectModelTo.eSet(modelToFeature, elementAttributeValue);
					}
				}
				
				modelTo.getContents().add(objectModelTo);
			}			
		}
		
		return modelTo;
	}
}
