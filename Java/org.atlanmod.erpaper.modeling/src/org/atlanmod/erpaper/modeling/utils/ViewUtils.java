package org.atlanmod.erpaper.modeling.utils;

import java.util.HashMap;
import java.util.Map;

import org.atlanmod.emfviews.virtuallinks.ConcreteConcept;
import org.atlanmod.emfviews.virtuallinks.ConcreteElement;
import org.atlanmod.emfviews.virtuallinks.ContributingModel;
import org.atlanmod.emfviews.virtuallinks.Filter;
import org.atlanmod.emfviews.virtuallinks.VirtualAssociation;
import org.atlanmod.emfviews.virtuallinks.VirtualLinksFactory;
import org.atlanmod.emfviews.virtuallinks.WeavingModel;
import org.eclipse.emf.ecore.EPackage;

public class ViewUtils {

	protected VirtualLinksFactory vLinksFactory;

	protected WeavingModel weavingModel;

	protected HashMap<String, ContributingModel> contributingModels;
	
	protected HashMap<String, ConcreteConcept> concreteConcepts;

	/**
	 * 
	 * @param vLinksFactory
	 */
	public ViewUtils(VirtualLinksFactory vLinksFactory) {
		this.vLinksFactory = vLinksFactory;
		this.contributingModels = new HashMap<String, ContributingModel>();
		this.concreteConcepts = new HashMap<String, ConcreteConcept>();
	}

	/**
	 * Get a concept from the concepts collections based on the URI
	 * 
	 * @param uri
	 * @return
	 */
	public ContributingModel getContributingModel(String uri) {
		return contributingModels.get(uri);
	}
	
	/**
	 * Set a contributing model in the View
	 * 
	 * @param uri
	 * @param contribModel
	 * @return
	 */
	public void setContributingModel(String uri, ContributingModel contribModel) {
		contributingModels.put(uri, contribModel);
	}
	
	/**
	 * Set all contributing models in the View
	 * 
	 * @param allContributingModels
	 * @return
	 */
	public void setContributingModels(HashMap<String, ContributingModel> allContributingModels) {
		contributingModels = allContributingModels;
	}
	
	public WeavingModel getWeavingModel() {
		return weavingModel;
	}

	/**
	 * Create and return a weaving model to be used to create a Viewpoint
	 * programmatically
	 * 
	 * @param name WeavingModel name
	 */
	public void createWeavingModel(String name, Boolean whitelist) {
		weavingModel = vLinksFactory.createWeavingModel();
		weavingModel.setName(name);
		weavingModel.setWhitelist(whitelist);
	}

	/**
	 * 
	 * @param nsURI
	 * @param path
	 * @return
	 */
	public ConcreteConcept createConcreteConcept(String nsURI, String path) {
		
		ContributingModel existentContributionModel = this.contributingModels.get(nsURI);
		
		
		
		ContributingModel contribModel = null;
		if (existentContributionModel == null) {
			contribModel = vLinksFactory.createContributingModel();
			weavingModel.getContributingModels().add(contribModel);
			contribModel.setURI(nsURI);
			contributingModels.put(nsURI, contribModel);
		} else {
			contribModel = existentContributionModel;
		}
			
		ConcreteElement concreteConcept = null;
		
		ConcreteConcept existentConcreteElement = this.concreteConcepts.get(nsURI+path);
		
		if (existentConcreteElement == null) {
			concreteConcept = vLinksFactory.createConcreteConcept();
			contribModel.getConcreteElements().add(concreteConcept);
			concreteConcept.setPath(path);
			concreteConcepts.put(nsURI+path, (ConcreteConcept) concreteConcept);
		} else {
			concreteConcept = existentConcreteElement;
		}			
		
		return (ConcreteConcept) concreteConcept;
	}
	
	public ConcreteConcept createConcreteConceptWith(String keyContribModel, String path) {

		ConcreteConcept concreteConcept = vLinksFactory.createConcreteConcept();

		ContributingModel contribModel = weavingModel.getContributingModels().get(0);

		contribModel.getConcreteElements().add(concreteConcept);
		concreteConcept.setPath(path);
		
		return concreteConcept;
	}

	/**
	 * 
	 * @param contribModelNsURI
	 * @param elementPath
	 * @return
	 */
	public ConcreteElement createConcreteElement(String contribModelNsURI, String elementPath) {

		ContributingModel contributingModel = this.getContributingModel(contribModelNsURI);

		ConcreteElement cElement = vLinksFactory.createConcreteElement();
		contributingModel.getConcreteElements().add(cElement);
		cElement.setPath(elementPath);
		
		return cElement;
	}

	/**
	 * 
	 * @param name
	 * @param source
	 * @param target
	 * @param upperBound
	 * @return
	 */
	public VirtualAssociation createVirtualAssociation(String name, ConcreteConcept source, ConcreteConcept target,
			int upperBound) {
		
		VirtualAssociation vAssociation = vLinksFactory.createVirtualAssociation();
		weavingModel.getVirtualLinks().add(vAssociation);
		vAssociation.setName(name);
		vAssociation.setUpperBound(upperBound);
		vAssociation.setSource(source);
		vAssociation.setTarget(target);
		
		return vAssociation;
	}

	/**
	 * 
	 * @param name
	 * @param target
	 * @return
	 */
	public Filter createFilter(String name, ConcreteElement target) {
		
		Filter filter = vLinksFactory.createFilter();
		weavingModel.getVirtualLinks().add(filter);
		filter.setName(name);
		filter.setTarget(target);
		
		return filter;
	}

	public void createContributingModels(Map<String, EPackage> viewpointContirbutingModelsURIs) {
		for (Map.Entry<String, EPackage> entry : viewpointContirbutingModelsURIs.entrySet()) {
			
			EPackage ePackage = entry.getValue();
			ContributingModel cm = vLinksFactory.createContributingModel();
			weavingModel.getContributingModels().add(cm);
			cm.setURI(ePackage.getNsURI());
			
			this.contributingModels.put(ePackage.getNsURI(), cm);			
		}	
	}
}
