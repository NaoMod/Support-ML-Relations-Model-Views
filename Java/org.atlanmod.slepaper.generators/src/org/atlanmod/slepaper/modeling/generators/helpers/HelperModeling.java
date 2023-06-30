package org.atlanmod.slepaper.modeling.generators.helpers;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;

import org.eclipse.emf.common.util.BasicEList;
import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EReference;
import org.eclipse.emf.ecore.EStructuralFeature;
import org.eclipse.emf.ecore.EcorePackage;

public class HelperModeling {
	
	/**
	 * Basic EObject generator based on the package and name
	 * 
	 * @param nsURI
	 * @param name
	 * @return
	 */
	public static EObject createEObject(String nsURI, String name) {
		EPackage ePackage = EPackage.Registry.INSTANCE.getEPackage(nsURI);
		EClass eClass = (EClass) ePackage.getEClassifier(name);
		return ePackage.getEFactoryInstance().create(eClass);
	}

	/**
	 * Create an object A of the package uri using random values
	 * 
	 * @return
	 */
	public static EObject createObjectTypeA(String uri, int id) {

		float randomAttr = RandomGen.generateFloat(1, 100000);
		String randomAttrS = RandomGen.getAlphaNumericString(RandomGen.generateInt(3, 15));

		EObject objA = createEObject(uri, "A");
		setAAttr(objA, id, randomAttr, randomAttrS);

		return objA;
	}

	/**
	 * Create an object A of the package uri using sent value
	 * 
	 * @return
	 */
	public static EObject createObjectTypeA(String uri, int id, float valueA, String valueS) {

		EObject objA = createEObject(uri, "A");
		setAAttr(objA, id, valueA, valueS);

		return objA;
	}

	/**
	 * Set all the parameter of objA of type A with the requested values
	 * 
	 * @param objA
	 * @param randomAttr
	 */
	private static void setAAttr(EObject objA, int ID, float randomAttr, String randomAttrS) {
		EStructuralFeature idFeature = objA.eClass().getEStructuralFeature("ID");
		if (idFeature != null && idFeature.getEType() == EcorePackage.Literals.EINT) {
			objA.eSet(idFeature, ID);
		}

		EStructuralFeature aFeature = objA.eClass().getEStructuralFeature("a");
		if (aFeature != null && aFeature.getEType() == EcorePackage.Literals.EFLOAT) {
			objA.eSet(aFeature, randomAttr);
		}

		EStructuralFeature sFeature = objA.eClass().getEStructuralFeature("s");
		if (sFeature != null && sFeature.getEType() == EcorePackage.Literals.ESTRING) {
			objA.eSet(sFeature, randomAttrS);
		}
	}

	/**
	 * Create an object B of the package http://AB using random values
	 * 
	 * @return
	 */
	public static EObject createObjectTypeB(String uri, int id) {

		float randomAttrB = RandomGen.generateFloat(1, 100000);
		float randomAttrC = RandomGen.generateFloat(1, 100000);
		float randomAttrD = RandomGen.generateFloat(1, 100000);
		String randomAttrS = RandomGen.getAlphaNumericString(RandomGen.generateInt(3, 15));

		EObject objB = createEObject(uri, "B");
		setBAttr(objB, id, randomAttrB, randomAttrC, randomAttrD, randomAttrS);

		return objB;
	}

	/**
	 * Create an object B of the package http://AB using random values with fixed
	 * value of attribute b
	 * 
	 * @return
	 */
	public static EObject createObjectTypeB(String uri, int id, float valueB) {

		float randomAttrC = RandomGen.generateFloat(1, 10000);
		float randomAttrD = RandomGen.generateFloat(1, 10000);
		String randomAttrS = RandomGen.getAlphaNumericString(RandomGen.generateInt(3, 15));

		EObject objB = createEObject(uri, "B");
		setBAttr(objB, id, valueB, randomAttrC, randomAttrD, randomAttrS);

		return objB;
	}

	/**
	 * Create an object B of the package http://AB using random values with fixed
	 * value of attribute b
	 * 
	 * @return
	 */
	public static EObject createObjectTypeB(String uri, int id, float valueB, String valueS) {

		float randomAttrC = RandomGen.generateFloat(1, 10000);
		float randomAttrD = RandomGen.generateFloat(1, 10000);

		EObject objB = createEObject(uri, "B");
		setBAttr(objB, id, valueB, randomAttrC, randomAttrD, valueS);

		return objB;
	}
	
	/**
	 * Create an object B of the package http://bn
	 * 
	 * @return
	 */
	public static EObject createObjectTypeBN(String uri, int id, List<String> types) {

		EObject objBN = createEObject(uri, "B");
		setBNAttr(objBN, id, types);

		return objBN;
	}

	/**
	 * Create an object B of the package http://AB using random values with fixed
	 * value of attribute b
	 * 
	 * @return
	 */
	public static EObject createObjectTypeB(String uri, int id, float valueB, float valueC, float valueD,
			String valueS) {

		EObject objB = createEObject(uri, "B");
		setBAttr(objB, id, valueB, valueC, valueD, valueS);

		return objB;
	}

	/**
	 * Set all the parameter of objB of type B with the requested values
	 * 
	 * @param objB
	 * @param randomAttrB
	 * @param randomAttrC
	 * @param randomAttrD
	 */
	private static void setBAttr(EObject objB, int ID, float randomAttrB, float randomAttrC, float randomAttrD,
			String randomAttrS) {
		EStructuralFeature idFeature = objB.eClass().getEStructuralFeature("ID");
		if (idFeature != null && idFeature.getEType() == EcorePackage.Literals.EINT) {
			objB.eSet(idFeature, ID);
		}

		EStructuralFeature bFeature = objB.eClass().getEStructuralFeature("b");
		if (bFeature != null && bFeature.getEType() == EcorePackage.Literals.EFLOAT) {
			objB.eSet(bFeature, randomAttrB);
		}

		EStructuralFeature cFeature = objB.eClass().getEStructuralFeature("c");
		if (cFeature != null && cFeature.getEType() == EcorePackage.Literals.EFLOAT) {
			objB.eSet(cFeature, randomAttrC);
		}

		EStructuralFeature dFeature = objB.eClass().getEStructuralFeature("d");
		if (dFeature != null && dFeature.getEType() == EcorePackage.Literals.EFLOAT) {
			objB.eSet(dFeature, randomAttrD);
		}

		EStructuralFeature sFeature = objB.eClass().getEStructuralFeature("s");
		if (sFeature != null && sFeature.getEType() == EcorePackage.Literals.ESTRING) {
			objB.eSet(sFeature, randomAttrS);
		}
	}
	
	/**
	 * Set all the parameter of objB of type B with the requested values
	 * 
	 * @param objBN
	 * @param randomAttrType
	 * @param randomAttrC
	 * @param randomAttrD
	 */
	private static void setBNAttr(EObject objBN, int ID, List<String> randomAttrType) {
		EStructuralFeature idFeature = objBN.eClass().getEStructuralFeature("ID");
		if (idFeature != null && idFeature.getEType() == EcorePackage.Literals.EINT) {
			objBN.eSet(idFeature, ID);
		}

		EStructuralFeature typeFeature = objBN.eClass().getEStructuralFeature("type");
		if (typeFeature != null && typeFeature.getEType() == EcorePackage.Literals.ESTRING) {
			EList<String> myList = new BasicEList<String>();
			
			for (String value : randomAttrType) {
			    myList.add((String) value);
			}
			
			objBN.eSet(typeFeature, myList);
		}
	}

	/**
	 * Create a link between A and B
	 * 
	 * @param objA
	 * @param objB
	 */
	public static void createLinkAB(EObject objA, EObject objB) {

		List<EReference> eReferences = objA.eClass().getEReferences();
		for (Iterator<EReference> iter = eReferences.iterator(); iter.hasNext();) {
			EReference ref = (EReference) iter.next();
			if (ref.isChangeable()) {
				objA.eSet(ref, objB);
			}
		}
	}
	
	/**
	 * Create a link between A and B
	 * 
	 * @param objA
	 * @param objB
	 */
	public static void createLinkCSV(EObject objA, EObject objB, String path) throws IOException{

		boolean fileExists = new File(path).exists();
        FileWriter csvWriter = new FileWriter(path, true);
        if (!fileExists) {
            csvWriter.append("A_id");
            csvWriter.append(",");
            csvWriter.append("B_id");
            csvWriter.append("\n");
        }
        EStructuralFeature aIdFeature = objA.eClass().getEStructuralFeature("ID");
        EStructuralFeature bIdFeature = objB.eClass().getEStructuralFeature("ID");
		if (aIdFeature != null && aIdFeature.getEType() == EcorePackage.Literals.EINT &&
			bIdFeature != null && bIdFeature.getEType() == EcorePackage.Literals.EINT) {
			csvWriter.append( objA.eGet(aIdFeature).toString());
			csvWriter.append(",");
			csvWriter.append( objB.eGet(bIdFeature).toString());
			csvWriter.append("\n");
		}        
        csvWriter.flush();
        csvWriter.close();
	}
	
	public static void createLinkCSV(Integer objAId, Integer objBId, String path) throws IOException{
        
		
        FileWriter csvWriter = new FileWriter(path, true);
        
		csvWriter.write(objAId.toString());
		csvWriter.write(",");
		csvWriter.write(objBId.toString());
		csvWriter.write("\n");
		       
        csvWriter.flush();
        csvWriter.close();
	}
	
	public static void createLinkCSVFile(String path) throws IOException{
		
		boolean fileExists = new File(path).exists();
        
		FileWriter csvWriter = null;
        if (fileExists) {
        	csvWriter = new FileWriter(path, false);
        } else {
        	csvWriter = new FileWriter(path, true);
        }
		
		csvWriter.append("A_id");
        csvWriter.append(",");
        csvWriter.append("B_id");
        csvWriter.append("\n");
        
        csvWriter.flush();
        csvWriter.close();
	}
}
