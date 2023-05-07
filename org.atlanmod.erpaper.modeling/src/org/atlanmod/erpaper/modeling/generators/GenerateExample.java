package org.atlanmod.erpaper.modeling.generators;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.atlanmod.erpaper.modeling.generators.helpers.HelperModeling;
import org.atlanmod.erpaper.modeling.generators.helpers.RandomGen;
import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.EcoreResourceFactoryImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;

public class GenerateExample {

	public static String here = new File(".").getAbsolutePath();

	public static URI resourceURI(String relativePath) {
		return URI.createFileURI(here + relativePath);
	}

	/**
	 * Method to generate two models selecting random values from the numeric and string sets defined
	 * The method also select a random number of links and missing links to be included in the generation of the dataset
	 * 
	 * 
	 * @param directory Name of the directory where the serialized models will be stored as xmi files
	 * @param IDs       Set of IDs to be used as identifiers of the elements
	 * @param numbers   Set of numbers to be selected as elements numerical attributes
	 * @param strings   Set of strings to be selected as elements string attributes
	 * 
	 * @throws IOException
	 */
	public static void generateDataset(String directory, Set<Integer> IDs, Set<Float> numbers,
			Set<String> strings) throws IOException {
		// Create EMF Resource for serialization
		ResourceSet rSet = new ResourceSetImpl();
		rSet.getResourceFactoryRegistry().getExtensionToFactoryMap().put("xmi", new XMIResourceFactoryImpl());

		Resource modelA = null;
		URI uriA;
		Resource modelB = null;
		URI uriB;

		Iterator<Float> itN = numbers.iterator();
		Iterator<String> itS = strings.iterator();
		Iterator<Integer> itId = IDs.iterator();
		
		List<Float> values = new ArrayList<Float>();
		
		Map<Float, Integer> idsa = new HashMap<Float, Integer>();
		
		// A.a = B.b
		uriA = resourceURI("/../Modeling_Resources/models/Example/" + directory + "/DatasetA.xmi");
		modelA = rSet.createResource(uriA);
		uriB = resourceURI("/../Modeling_Resources/models/Example/" + directory + "/DatasetB.xmi");
		modelB = rSet.createResource(uriB);
		
		for (int i = 0; i <= 609; i++) {

			Float value = itN.next();
			itN.remove();

			String ASValue = itS.next();
			itS.remove();
			
			Integer id = itId.next();

			EObject objectA = HelperModeling.createObjectTypeA("http://a_model", id,  value, ASValue);
			
			modelA.getContents().add(objectA);
			
			values.add(value);
			idsa.put(value, id);
			
			modelA.getContents().add(objectA);
		}
		Collections.sort(values);
		Iterator<Float> vIt = values.iterator();
		int linksQuantity = 100836;
		int bquantity = 9741;
		
		String csvPath = here + "/../Modeling_Resources/models/Example/" + directory + "/Relations.csv";
		HelperModeling.createLinkCSVFile(csvPath);			
		
		for (int i = 0; i <= 609; i++) {
			
			Float avalue = vIt.next();
			vIt.remove();
			Integer aId = idsa.get(avalue);
			
			
			if (bquantity > 0 && linksQuantity > 0) {
				//random int to decide the number of links
				int howManyLinks = RandomGen.generateInt(1, 10);
				int howManyMissingLinks = RandomGen.generateInt(1, 10);
				
				for (int j = 0; j <= howManyLinks; j++) {					
					
					String BSValue = itS.next();
					itS.remove();
					
					Integer idb = itId.next();	

					EObject objectB = HelperModeling.createObjectTypeB("http://b_model", idb,  avalue, BSValue);
					
					modelB.getContents().add(objectB);
					
					HelperModeling.createLinkCSV(aId, idb, csvPath);
					
					
					linksQuantity--;
					bquantity--;
				}
				
				for (int k = 0; k <= howManyMissingLinks; k++) {					
					
					
					String BSValue = itS.next();
					itS.remove();
					
					Integer idb = itId.next();	

					EObject objectB = HelperModeling.createObjectTypeB("http://b_model", idb,  avalue, BSValue);
					
					modelB.getContents().add(objectB);
					bquantity--;
				}
			}			
		}

		// serialize
		try {
			modelA.save(null);
			modelB.save(null);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/***
	 * 
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {
		// Create basic resources to deal with EMF reflective API
		Map<String, Object> map = Resource.Factory.Registry.INSTANCE.getExtensionToFactoryMap();
		map.put("xmi", new XMIResourceFactoryImpl());
		map.put("ecore", new EcoreResourceFactoryImpl());

		// Create EMF Resources and register metamodels used in the example
		ResourceSet rs = new ResourceSetImpl();
		EPackage APkg = (EPackage) rs.getResource(resourceURI("/../Modeling_Resources/metamodels/A.ecore"), true).getContents().get(0);
		EPackage.Registry.INSTANCE.put(APkg.getNsURI(), APkg);
		EPackage BPkg = (EPackage) rs.getResource(resourceURI("/../Modeling_Resources/metamodels/B.ecore"), true).getContents().get(0);
		EPackage.Registry.INSTANCE.put(BPkg.getNsURI(), BPkg);

		//Create Sets of numbers to be used in the generated models
		Set<Float> numberList = RandomGen.createRandomFloats(1000 * 10, 100);
		Set<String> stringsList = RandomGen.createRandomStrings(2000 * 20);
		Set<Integer> IdList = RandomGen.createRandomIds(2000 * 10);
		
		// Generate the dataset as XMI files and CSV relation into Data/AB directory
		generateDataset("AB", IdList, numberList, stringsList);
	}
}
