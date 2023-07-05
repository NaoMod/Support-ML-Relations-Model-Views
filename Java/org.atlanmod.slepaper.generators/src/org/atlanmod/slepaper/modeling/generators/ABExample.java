package org.atlanmod.slepaper.modeling.generators;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.IntStream;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.atlanmod.slepaper.modeling.generators.helpers.HelperModeling;
import org.atlanmod.slepaper.modeling.generators.helpers.RandomGen;
import org.atlanmod.slepaper.modeling.generators.helpers.UserMovies;
import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EStructuralFeature;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.EcoreResourceFactoryImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;

public class ABExample {
	public static String here = new File(".").getAbsolutePath();

	public static URI resourceURI(String relativePath) {
		return URI.createFileURI(here + relativePath);
	}

	/**
	 * Method to generate the models A and B for each rule using random values
	 * 
	 * 
	 * @param directory  Name of the directory where the serialized models will be
	 *                   stored as xmi files
	 * @param IDs        Set of IDs to be used as identifiers of the elements
	 * @param numbers    Set of numbers to be selected as elements numerical
	 *                   attributes
	 * @param strings    Set of strings to be selected as elements string attributes
	 * @param ruleNumber Number identifier for the rule
	 * 
	 * @throws IOException
	 */
	public static void generateDataset(String directory, Set<Integer> IDs, Set<Float> numbers, Set<String> strings,
			int ruleNumber) throws IOException {
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
		List<Float[]> valuesM = new ArrayList<Float[]>();
		List<String> valuesS = new ArrayList<String>();

		Map<Float, Integer> idsa = new HashMap<Float, Integer>();
		Map<String, Integer> idsaS = new HashMap<String, Integer>();

		Iterator<Float> vIt;
		Iterator<Float[]> vItM;
		Iterator<String> vItS;
		
		int linksQuantity;
		int bquantity;

		String csvPath;

		switch (ruleNumber) {
		case 1:
			// A.a = B.b
			uriA = resourceURI("/../../Modeling_Resources/models/Example/" + directory + ruleNumber + "/DatasetA.xmi");
			modelA = rSet.createResource(uriA);
			uriB = resourceURI("/../../Modeling_Resources/models/Example/" + directory + ruleNumber + "/DatasetB.xmi");
			modelB = rSet.createResource(uriB);

			for (int i = 1; i <= 1000; i++) {

				Float value = itN.next();
				itN.remove();

				String ASValue = itS.next();
				itS.remove();

				Integer id = itId.next();

				EObject objectA = HelperModeling.createObjectTypeA("http://a_model", id, value, ASValue);

				modelA.getContents().add(objectA);

				values.add(value);
				idsa.put(value, id);

				modelA.getContents().add(objectA);
			}

			Collections.sort(values);
			vIt = values.iterator();
			linksQuantity = 5000;
			bquantity = 3000;

			csvPath = here + "/../../Modeling_Resources/models/Example/" + directory + ruleNumber + "/Relations.csv";
			HelperModeling.createLinkCSVFile(csvPath);
			for (int i = 1; i <= bquantity; i++) {

				Float avalue = vIt.next();
				vIt.remove();
				Integer aId = idsa.get(avalue);

				if (bquantity > 0 && linksQuantity > 0) {
					// random int to decide the number of links
					int howManyLinks = RandomGen.generateInt(1, 10);
					int howManyMissingLinks = RandomGen.generateInt(1, 10);

					for (int j = 0; j <= howManyLinks; j++) {

						String BSValue = itS.next();
						itS.remove();

						Integer idb = itId.next();

						EObject objectB = HelperModeling.createObjectTypeB("http://b_model", idb, avalue, BSValue);

						modelB.getContents().add(objectB);

						HelperModeling.createLinkCSV(aId, idb, csvPath);

						linksQuantity--;
						bquantity--;
					}

					for (int k = 0; k <= howManyMissingLinks; k++) {

						String BSValue = itS.next();
						itS.remove();

						Integer idb = itId.next();

						EObject objectB = HelperModeling.createObjectTypeB("http://b_model", idb, avalue, BSValue);

						modelB.getContents().add(objectB);
						bquantity--;
					}
				}
			}
			break;
		case 2:
			// A.a = B.c * B.d
			uriA = resourceURI("/../../Modeling_Resources/models/Example/" + directory + ruleNumber + "/DatasetA.xmi");
			modelA = rSet.createResource(uriA);
			uriB = resourceURI("/../../Modeling_Resources/models/Example/" + directory + ruleNumber + "/DatasetB.xmi");
			modelB = rSet.createResource(uriB);

			for (int i = 1; i <= 1000; i++) {

				Float valueC = itN.next();
				itN.remove();
				Float valueD = itN.next();
				itN.remove();

				String ASValue = itS.next();
				itS.remove();

				Integer id = itId.next();

				EObject objectA = HelperModeling.createObjectTypeA("http://a_model", id, valueC * valueD, ASValue);

				modelA.getContents().add(objectA);

				Float[] arr = new Float[2];
				arr[0] = valueC;
				arr[1] = valueD;

				valuesM.add(arr);
				idsa.put(valueC * valueD, id);

				modelA.getContents().add(objectA);
			}

			vItM = valuesM.iterator();
			linksQuantity = 5000;
			bquantity = 3000;

			csvPath = here + "/../../Modeling_Resources/models/Example/" + directory + ruleNumber + "/Relations.csv";
			HelperModeling.createLinkCSVFile(csvPath);
			for (int i = 1; i <= bquantity; i++) {

				Float[] cdvalue = vItM.next();
				vItM.remove();
				Float valueB = itN.next();
				itN.remove();
				Integer aId = idsa.get(cdvalue[0] * cdvalue[1]);

				if (bquantity > 0 && linksQuantity > 0) {
					// random int to decide the number of links
					int howManyLinks = RandomGen.generateInt(1, 10);
					int howManyMissingLinks = RandomGen.generateInt(1, 10);

					for (int j = 0; j <= howManyLinks; j++) {

						String BSValue = itS.next();
						itS.remove();

						Integer idb = itId.next();

						EObject objectB = HelperModeling.createObjectTypeB("http://b_model", idb, valueB, cdvalue[0],
								cdvalue[1], BSValue);

						modelB.getContents().add(objectB);

						HelperModeling.createLinkCSV(aId, idb, csvPath);

						linksQuantity--;
						bquantity--;
					}

					for (int k = 0; k <= howManyMissingLinks; k++) {

						String BSValue = itS.next();
						itS.remove();

						Integer idb = itId.next();

						EObject objectB = HelperModeling.createObjectTypeB("http://b_model", idb, cdvalue[0],
								cdvalue[1], valueB, BSValue);

						modelB.getContents().add(objectB);
						bquantity--;
					}
				}
			}
			break;
		case 3:
			// A.s.contains(B.s)
			uriA = resourceURI("/../../Modeling_Resources/models/Example/" + directory + ruleNumber + "/DatasetA.xmi");
			modelA = rSet.createResource(uriA);
			uriB = resourceURI("/../../Modeling_Resources/models/Example/" + directory + ruleNumber + "/DatasetB.xmi");
			modelB = rSet.createResource(uriB);

			for (int i = 1; i <= 1000; i++) {

				Float value = itN.next();
				itN.remove();

				String ASValue = itS.next();
				itS.remove();

				Integer id = itId.next();

				EObject objectA = HelperModeling.createObjectTypeA("http://a_model", id, value, ASValue);

				modelA.getContents().add(objectA);

				valuesS.add(ASValue);
				idsaS.put(ASValue, id);

				modelA.getContents().add(objectA);
			}

			Collections.sort(valuesS);
			vItS = valuesS.iterator();
			linksQuantity = 5000;
			bquantity = 3000;

			csvPath = here + "/../../Modeling_Resources/models/Example/" + directory + ruleNumber + "/Relations.csv";
			HelperModeling.createLinkCSVFile(csvPath);
			for (int i = 1; i <= bquantity; i++) {

				String aSvalue = vItS.next();
				vItS.remove();
				Integer aId = idsaS.get(aSvalue);

				if (bquantity > 0 && linksQuantity > 0) {
					// random int to decide the number of links
					int howManyLinks = RandomGen.generateInt(1, 10);
					int howManyMissingLinks = RandomGen.generateInt(1, 10);
					
					int sizeOfSubstring = RandomGen.generateInt(1, aSvalue.length());
					int randomNum = new Random().nextInt(aSvalue.length() - sizeOfSubstring + 1);
					String BSValue = aSvalue.substring(randomNum, randomNum + sizeOfSubstring);

					for (int j = 0; j <= howManyLinks; j++) {

						Float value = itN.next();
						itN.remove();

						Integer idb = itId.next();

						EObject objectB = HelperModeling.createObjectTypeB("http://b_model", idb, value, BSValue);

						modelB.getContents().add(objectB);

						HelperModeling.createLinkCSV(aId, idb, csvPath);

						linksQuantity--;
						bquantity--;
					}

					for (int k = 0; k <= howManyMissingLinks; k++) {

						Float value = itN.next();
						itN.remove();

						Integer idb = itId.next();

						EObject objectB = HelperModeling.createObjectTypeB("http://b_model", idb, value, BSValue);

						modelB.getContents().add(objectB);
						bquantity--;
					}
				}
			}
			break;
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
		EPackage APkg = (EPackage) rs.getResource(resourceURI("/../../Modeling_Resources/metamodels/Left.ecore"), true)
				.getContents().get(0);
		EPackage.Registry.INSTANCE.put(APkg.getNsURI(), APkg);
		EPackage BPkg = (EPackage) rs.getResource(resourceURI("/../../Modeling_Resources/metamodels/Right.ecore"), true)
				.getContents().get(0);
		EPackage.Registry.INSTANCE.put(BPkg.getNsURI(), BPkg);

		// Create Sets of numbers and strings to be used in the generated models
		Set<Float> numberList = RandomGen.createRandomFloats(1000 * 10, 100);
		Set<String> stringsList = RandomGen.createRandomStrings(2000 * 20);
		Set<Integer> IdList = RandomGen.createRandomIds(2000 * 10);

		// Generate the dataset as XMI files and CSV relation into Example/AB directory
		generateDataset("AB", IdList, numberList, stringsList, 3);
	}
}
