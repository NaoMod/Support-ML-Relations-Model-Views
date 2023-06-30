package org.atlanmod.slepaper.modeling.generators.helpers;

import java.util.Iterator;
import java.util.List;

import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EAttribute;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EReference;
import org.eclipse.emf.ecore.EStructuralFeature;
import org.eclipse.emf.ecore.EcorePackage;
import org.eclipse.emf.ecore.resource.Resource;

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
	 * Create an object Movie of the package uri and set the requested values,
	 * returning the object
	 * 
	 * @return
	 */
	public static EObject createObjectTypeMovie(String uri, int id, String title, String genres) {

		EObject objMovie = createEObject(uri, "Movie");

		EStructuralFeature idFeature = objMovie.eClass().getEStructuralFeature("id");

		if (idFeature != null && idFeature.getEType() == EcorePackage.Literals.EINT) {
			objMovie.eSet(idFeature, id);
		}

		EStructuralFeature titleFeature = objMovie.eClass().getEStructuralFeature("title");
		if (titleFeature != null && titleFeature.getEType() == EcorePackage.Literals.ESTRING) {
			objMovie.eSet(titleFeature, title);
		}

		return objMovie;
	}

	public static EObject createObjectTypeGenre(String uri, Integer id, String genre) {
		
		EObject objGenre = createEObject(uri, "Genre");

		EStructuralFeature idFeature = objGenre.eClass().getEStructuralFeature("id");

		if (idFeature != null && idFeature.getEType() == EcorePackage.Literals.EINT) {
			objGenre.eSet(idFeature, id);
		}

		EStructuralFeature valueFeature = objGenre.eClass().getEStructuralFeature("value");
		if (valueFeature != null && valueFeature.getEType() == EcorePackage.Literals.ESTRING) {
			objGenre.eSet(valueFeature, genre);
		}

		return objGenre;
	}

	/**
	 * Create an object User of the package uri and set the requested values,
	 * returning the object
	 * 
	 * @return
	 */
	public static EObject createObjectTypeUser(String uri, Integer userID, String name) {

		EObject objUser = createEObject(uri, "User");

		EStructuralFeature idFeature = objUser.eClass().getEStructuralFeature("id");

		if (idFeature != null && idFeature.getEType() == EcorePackage.Literals.EINT) {
			objUser.eSet(idFeature, userID);
		}

		EStructuralFeature nameFeature = objUser.eClass().getEStructuralFeature("name");
		if (nameFeature != null && nameFeature.getEType() == EcorePackage.Literals.ESTRING) {
			objUser.eSet(nameFeature, name);
		}

		return objUser;
	}

	public static EObject findObjectType(Resource model, String className, Integer identifier) {
		List<EObject> elements = model.getContents();

		for (Iterator<EObject> iter = elements.iterator(); iter.hasNext();) {
			EObject element = iter.next();
			EClass modelClass = element.eClass();

			if (modelClass.getName().equals(className)) {
				for (Iterator<EAttribute> iterAttr = modelClass.getEAllAttributes().iterator(); iterAttr.hasNext();) {
					EAttribute elementAttribute = (EAttribute) iterAttr.next();
					String attrName = elementAttribute.getName();

					if (attrName.equals("id")) {
						Object elementAttributeValue = element.eGet(elementAttribute);

						if (elementAttributeValue.equals(identifier)) {
							return element;
						}
					}
				}
			}
		}
		return null;
	}

	/**
	 * Create a link between User and Movie using rate relation
	 * 
	 * @param user
	 * @param movie
	 */
	public static void createRelation(EObject user, EObject movie) {

		List<EReference> eReferences = user.eClass().getEReferences();
		for (Iterator<EReference> iter = eReferences.iterator(); iter.hasNext();) {
			EReference ref = (EReference) iter.next();
			if (ref.getName().equals("movies")) {
				if (ref.isChangeable()) {
					@SuppressWarnings("unchecked")
					EList<EObject> rates = (EList<EObject>) user.eGet(ref);
					rates.add(movie);
				}
			}
		}
	}

	/**
	 * Create a link between Movie and genre using genres relation
	 * 
	 * @param movie
	 * @param genre
	 */
	public static void createGenreRelation(EObject movie, EObject genre) {

		List<EReference> eReferences = movie.eClass().getEReferences();
		for (Iterator<EReference> iter = eReferences.iterator(); iter.hasNext();) {
			EReference ref = (EReference) iter.next();
			if (ref.getName().equals("genres")) {
				if (ref.isChangeable()) {
					@SuppressWarnings("unchecked")
					EList<EObject> genres = (EList<EObject>) movie.eGet(ref);
					genres.add(genre);
				}
			}
		}
	}
}
