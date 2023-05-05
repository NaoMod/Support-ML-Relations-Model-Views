package org.atlanmod.erpaper.modeling.generators;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import org.atlanmod.emfviews.core.Viewpoint;
import org.atlanmod.emfviews.core.ViewpointResource;
import org.atlanmod.emfviews.virtuallinks.VirtualLinksPackage;
import org.atlanmod.erpaper.modeling.utils.EmfViewsFactory;
import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EAttribute;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EClassifier;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EReference;
import org.eclipse.emf.ecore.EStructuralFeature;
import org.eclipse.emf.ecore.impl.DynamicEObjectImpl;
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

	public static void main(String[] args) throws Exception {
		
		//Paths for Example View files
		String predictedViewDirectory = "/../Views/Predicted_View/";
		String parametersFile = predictedViewDirectory + "my_view/parameters.gnn";
		String eViewFile = predictedViewDirectory + "my_view/predicted.eview";
		String eViewPointFile = predictedViewDirectory + "src-gen/predicted.eviewpoint";
		String viewpointWMFile = predictedViewDirectory + "src-gen/predicted.xmi";
		
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
		EPackage.Registry baseRegistry = rSet.getPackageRegistry();
		
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
		Map<String, String> viewContribModels = new HashMap<String, String>();
		String[] aliasPathPairLeft = viewContribModelsString[0].split("::");
		String classLeft = aliasPathPairLeft[0];
		String modelTrainingLeft = aliasPathPairLeft[1].replace("../../", "/../");
		
		String[] aliasPathPairRight = viewContribModelsString[1].split("::");
		String classRight = aliasPathPairRight[0];
		String modelTrainingRight = aliasPathPairRight[1].replace("../../", "/../");
		
		//get VirtualAssociation
		//getVirtualAssociation(viewpointWM.getContents());	
		
		Properties props = new Properties();
		props.load(new FileReader(here + parametersFile));
		
		URI uriLeft = resourceURI(predictedViewDirectory + "/for_predict/DatasetLeft.xmi");
		Resource modelLeft = rSet.createResource(uriLeft);
		EPackage ePackageLeft = EPackage.Registry.INSTANCE.getEPackage(viewPointContribEPackages.get(classLeft).getNsURI());
		URI uriRight = resourceURI(predictedViewDirectory + "/for_predict/DatasetRight.xmi");
		EPackage ePackageRight = EPackage.Registry.INSTANCE.getEPackage(viewPointContribEPackages.get(classRight).getNsURI());
		Resource modelRight = rSet.createResource(uriRight);

		// TODO: Encapsulate in a for loop when have more models (how to deal with IDs?)
		URI uriTrainingLeft = resourceURI(modelTrainingLeft);
		Resource modelForTrainingLeft = rSet.getResource(uriTrainingLeft, true);
		URI uriTrainingRight = resourceURI(modelTrainingRight);
		Resource modelForTrainingRight = rSet.getResource(uriTrainingRight, true);

		copyModel(modelForTrainingLeft, modelLeft, ePackageLeft, classLeft);
		copyModel(modelForTrainingRight, modelRight, ePackageRight, classRight);
		
		String csvPath = here + predictedViewDirectory + "/for_predict/Relations.csv";
		
		createCSVSkeleton(csvPath, "Left_id", "Right_id");
		
		String jsonPath = here + predictedViewDirectory + "/for_predict/Parameters.json";
		
		props.put("CLASS_LEFT", classLeft);
		props.put("CLASS_RIGHT", classRight);
		createJsonParams(jsonPath, props);

		// serialize XMI files
		try {
			modelLeft.save(null);
			modelRight.save(null);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@SuppressWarnings("resource")
	private static void createJsonParams(String jsonPath, Properties props) throws IOException {
		Gson gsonObj = new Gson();
		String strJson =  gsonObj.toJson(props);
		
		FileWriter jsonWriter = new FileWriter(jsonPath, false);
		
		jsonWriter.append(strJson);
		
		
		jsonWriter.flush();
		jsonWriter.close();
		
	}

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

	public static Resource copyModel(Resource modelFrom, Resource modelTo, EPackage ePackageTo, String className) {
		//TODO: Include parameter to exclude attributes
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
	
	private static void getVirtualAssociation(EList<EObject> WMvElements) {
		for (Iterator<EObject> iter = WMvElements.iterator(); iter.hasNext();) {
			EObject vElement = iter.next();
			EClass vElementModelClass = vElement.eClass();
			String vElementModelClassName = vElementModelClass.getName();
			if(vElementModelClass.isInstance(VirtualLinksPackage.VIRTUAL_ASSOCIATION)) {
				System.out.println(vElementModelClassName);
			}
//			if(vElementModelClassName.equals(""))
//			if (compA.isInstance(vElement)) {
//				EClass AElementModelClass = vElement.eClass();
//				EStructuralFeature AIDFeature = AElementModelClass.getEStructuralFeature("ID");
//				EStructuralFeature aFeature = AElementModelClass.getEStructuralFeature("a");
//				EStructuralFeature ASFeature = AElementModelClass.getEStructuralFeature("s");
//
//				Float aFeatureValue = (Float) vElement.eGet(aFeature);
//				Integer AIDFeatureValue = (Integer) vElement.eGet(AIDFeature);
//				String ASFeatureValue = (String) vElement.eGet(ASFeature);
//				System.out.println(AIDFeatureValue);
//				List<EReference> eReferences = vElementModelClass.getEReferences();
//				for (Iterator<EReference> iterRef = eReferences.iterator(); iterRef.hasNext();) {
//					EReference ref = (EReference) iterRef.next();
//					@SuppressWarnings("unchecked")
//					EList<EObject> refElements = (EList<EObject>) vElement.eGet(ref);
//					if (!refElements.isEmpty()) {
//						EObject refElement = (EObject) refElements.get(0);
//						EClass refElementModelClass = refElement.eClass();
//						if (compB.isInstance(refElements.get(0))) {
//
//							EStructuralFeature BIDFeature = refElementModelClass.getEStructuralFeature("ID");
//							EStructuralFeature bFeature = refElementModelClass.getEStructuralFeature("b");
//							EStructuralFeature cFeature = refElementModelClass.getEStructuralFeature("c");
//							EStructuralFeature dFeature = refElementModelClass.getEStructuralFeature("d");
//							EStructuralFeature BSFeature = refElementModelClass.getEStructuralFeature("s");
//
//							Integer BIDFeatureValue = (Integer) refElement.eGet(BIDFeature);
//							Float bFeatureValue = (Float) refElement.eGet(bFeature);
//							Float cFeatureValue = (Float) refElement.eGet(cFeature);
//							Float dFeatureValue = (Float) refElement.eGet(dFeature);
//							String BSFeatureValue = (String) refElement.eGet(BSFeature);
//							
//							System.out.println(BIDFeatureValue);
//
//							ceateObjectExample(serialized, AIDFeatureValue, BIDFeatureValue, aFeatureValue,
//									bFeatureValue, cFeatureValue, dFeatureValue, ASFeatureValue, BSFeatureValue);
//						}
//					}
//
//				}
//			}
//			System.out.println(i++);
//
		}
//
//		try {
//			serialized.save(null);
//		} catch (IOException e) {
//			e.printStackTrace();
//		}
	}
}
