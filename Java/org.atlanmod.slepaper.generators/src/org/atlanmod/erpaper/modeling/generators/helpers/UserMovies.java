package org.atlanmod.erpaper.modeling.generators.helpers;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EStructuralFeature;
import org.eclipse.emf.ecore.EcorePackage;

public class UserMovies {
	
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
	 * Create an object Movie of the package uri and set the requested values, returning the object
	 * 
	 * @return
	 */
	public static EObject createObjectTypeMovie(String uri, int id, String title, String genres) {

		EObject objMovie = createEObject(uri, "Movie");
		
		EStructuralFeature idFeature = objMovie.eClass().getEStructuralFeature("ID");
		
		if (idFeature != null && idFeature.getEType() == EcorePackage.Literals.EINT) {
			objMovie.eSet(idFeature, id);
		}

		EStructuralFeature titleFeature = objMovie.eClass().getEStructuralFeature("Title");
		if (titleFeature != null && titleFeature.getEType() == EcorePackage.Literals.ESTRING) {
			objMovie.eSet(titleFeature, title);
		}

		EStructuralFeature genresFeature = objMovie.eClass().getEStructuralFeature("Genres");
		if (genresFeature != null && genresFeature.getEType() == EcorePackage.Literals.ESTRING) {
			objMovie.eSet(genresFeature, genres);
		}

		return objMovie;
	}

	/**
	 * Create an object User of the package uri and set the requested values, returning the object
	 * 
	 * @return
	 */
	public static EObject createObjectTypeUser(String uri, Integer userID, String name) {
		
		EObject objUser = createEObject(uri, "User");
		
		EStructuralFeature idFeature = objUser.eClass().getEStructuralFeature("ID");
		
		if (idFeature != null && idFeature.getEType() == EcorePackage.Literals.EINT) {
			objUser.eSet(idFeature, userID);
		}

		EStructuralFeature nameFeature = objUser.eClass().getEStructuralFeature("Name");
		if (nameFeature != null && nameFeature.getEType() == EcorePackage.Literals.ESTRING) {
			objUser.eSet(nameFeature, name);
		}

		return objUser;
	}
}
