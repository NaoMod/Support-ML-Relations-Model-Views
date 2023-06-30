package org.atlanmod.slepaper.modeling.generators;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.atlanmod.slepaper.modeling.generators.helpers.RandomGen;
import org.atlanmod.slepaper.modeling.generators.helpers.UserMovies;
import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.EcoreResourceFactoryImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;

public class UserMoviesExample {
	public static String here = new File(".").getAbsolutePath();

	public static URI resourceURI(String relativePath) {
		return URI.createFileURI(here + relativePath);
	}

	/**
	 * Method to generate the model copying the values from the ml-latest-small
	 * movielens dataset
	 * 
	 * 
	 * @param directory Name of the directory where the serialized models will be
	 *                  stored as xmi files
	 * @param strings   Set of strings to be used as user names (the dataset does
	 *                  not contain any info about users)
	 * 
	 * @throws IOException
	 */
	public static void generateDataset(String directory, Set<String> names) throws IOException {
		// Create EMF Resource for serialization
		ResourceSet rSet = new ResourceSetImpl();
		rSet.getResourceFactoryRegistry().getExtensionToFactoryMap().put("xmi", new XMIResourceFactoryImpl());

		Resource model = null;
		URI uriModel;

		Iterator<String> itNames = names.iterator();

		uriModel = resourceURI("/../../Modeling_Resources/models/Example/" + directory + "/UsersMovies.xmi");
		model = rSet.createResource(uriModel);

		String pathToCsvMovies = here + "/ml-latest-small/movies.csv";
		BufferedReader csvReaderMovie = new BufferedReader(new FileReader(pathToCsvMovies));
		
		Map<String, Integer> uniqueGenres = new HashMap<String, Integer>();

		CSVParser parser = CSVParser.parse(csvReaderMovie, CSVFormat.RFC4180);
		for (CSVRecord csvRecord : parser) {
			try {
				Integer movieID = Integer.parseInt(csvRecord.get(0));
				String title = csvRecord.get(1);
				String genres = csvRecord.get(2);

				EObject objectMovie = UserMovies.createObjectTypeMovie("http://paper/movies", movieID, title, genres);
				
				model.getContents().add(objectMovie);
				
				String [] genresSplit = genres.split("\\|");
				for (String genre : genresSplit) {
					
					EObject objectGenre;
					if (!uniqueGenres.containsKey(genre)) {
						
						Integer id = RandomGen.generateID();
	
						objectGenre = UserMovies.createObjectTypeGenre("http://paper/movies", id, genre);
		
						model.getContents().add(objectGenre);
						uniqueGenres.put(genre, id);
					} else {
						Integer id = uniqueGenres.get(genre);
						objectGenre = UserMovies.findObjectType(model, "Genre", id);
					}
					
					//create the rate relation
					UserMovies.createGenreRelation(objectMovie, objectGenre);
				}

				
			} catch (NumberFormatException e) {
				continue;
			}
		}

		String pathToCsvUsers = here + "/ml-latest-small/ratings.csv";
		BufferedReader csvReaderUser = new BufferedReader(new FileReader(pathToCsvUsers));

		CSVParser parserUsers = CSVParser.parse(csvReaderUser, CSVFormat.RFC4180);
		List<Integer> uniqueUsers = new ArrayList<Integer>();
		for (CSVRecord csvRecord : parserUsers) {
			try {
				Integer userID = Integer.parseInt(csvRecord.get(0));
				Integer movieID = Integer.parseInt(csvRecord.get(1));
				EObject objectUser;
				if (!uniqueUsers.contains(userID)) {
					String name = itNames.next();
	
					objectUser = UserMovies.createObjectTypeUser("http://paper/users", userID, name);
	
					model.getContents().add(objectUser);
					uniqueUsers.add(userID);
				} else {
					objectUser = UserMovies.findObjectType(model, "User", userID);
				}
				EObject objectMovie = UserMovies.findObjectType(model, "Movie", movieID);
				//create the movies relation
				UserMovies.createRelation(objectUser, objectMovie);
			} catch (NumberFormatException e) {
				continue;
			}
		}

		// serialize
		try {
			model.save(null);
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
		

		EPackage UserMoviesPkg = (EPackage) rs
				.getResource(resourceURI("/../../Modeling_Resources/metamodels/UserMovies.ecore"), true).getContents()
				.get(0);
		EPackage.Registry.INSTANCE.put(UserMoviesPkg.getNsURI(), UserMoviesPkg);
		
		//register inner packages
		List<EObject> contents = UserMoviesPkg.eContents();

		// Iterate through the contents to find the packages
		for (EObject content : contents) {
		  if (content instanceof EPackage) {
		    EPackage pkg = (EPackage) content;
		    
		    EPackage.Registry.INSTANCE.put(pkg.getNsURI(), pkg);
		  }
		}

		// Create Sets of strings to be used as user names in the generated models
		Set<String> stringsList = RandomGen.createRandomStrings(700);

		// Generate the dataset as XMI files
		generateDataset("MovieLens", stringsList);
	}
}
