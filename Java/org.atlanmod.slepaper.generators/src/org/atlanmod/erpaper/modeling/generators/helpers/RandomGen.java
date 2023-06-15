package org.atlanmod.erpaper.modeling.generators.helpers;

import java.nio.charset.Charset;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/**
 * Class used to generate random numbers and strings to be used in the generated models
 *
 */
public class RandomGen {
	
	/**
	 * Adapted from
	 * https://www.geeksforgeeks.org/generate-random-string-of-given-size-in-java/
	 * 
	 * @param n
	 * @return
	 */
	public static String getAlphaNumericString(int n) {

		byte[] array = new byte[256];
		new Random().nextBytes(array);

		String randomString = new String(array, Charset.forName("UTF-8"));

		StringBuffer r = new StringBuffer();

		for (int k = 0; k < randomString.length(); k++) {

			char ch = randomString.charAt(k);

			if (((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')) && (n > 0)) {

				r.append(ch);
				n--;
			}
		}

		return r.toString();
	}

	/**
	 * Generate a random all caps string without space
	 * 
	 * @param n
	 * @return
	 */
	public static String getRandomStringUpper(int n) {

		byte[] array = new byte[256];
		new Random().nextBytes(array);

		String randomString = new String(array, Charset.forName("UTF-8"));

		StringBuffer r = new StringBuffer();

		for (int k = 0; k < randomString.length(); k++) {

			char ch = randomString.charAt(k);

			if (((ch >= 'A' && ch <= 'Z')) && (n > 0)) {

				r.append(ch);
				n--;
			}
		}

		return r.toString();
	}

	/**
	 * Randomly create random int between min and max parameters
	 * 
	 * @return
	 */
	public static int generateInt(int min, int max) {
		int randomNum = new Random().nextInt(max - min) + min;
		return randomNum;
	}

	/**
	 * Randomly create IDs for an element
	 * 
	 * @return
	 */
	public static int generateID() {
		int randomID = new Random().nextInt(50000 - 1) + 1;
		return randomID;
	}

	/**
	 * Randomly get float number
	 * 
	 * @return
	 */
	public static float generateFloat(float min, float max) {
		Random r = new Random();
		float randoD = min + r.nextFloat() * (max - min);
		return randoD;
	}

	/**
	 * Utility function that creates a set of randomly generated floats of
	 * specified size. We use a Set to avoid duplicates.
	 */
	public static Set<Float> createRandomFloats(int sz, int limit) {
		Set<Float> ret = new HashSet<Float>();
		
		while (ret.size() < sz) {
			ret.add(generateFloat(1, limit));
		}

		return ret;
	}
	
	/**
	 * Utility function that creates a set of randomly generated strings of
	 * specified size. We use a Set to avoid duplicates.
	 */
	public static Set<String> createRandomStrings(int sz) {
		Set<String> ret = new HashSet<String>();

		while (ret.size() < sz) {
			//strings with size between 3 and 15
			ret.add(getAlphaNumericString(generateInt(3, 15)));
		}

		return ret;
	}
	
	/**
	 * Utility function that creates a set of randomly generated IDs of
	 * It uses a Set to avoid duplicates.
	 */
	public static Set<Integer> createRandomIds(int sz) {
		Set<Integer> ret = new HashSet<Integer>();

		while (ret.size() < sz) {
			ret.add(generateID());
		}

		return ret;
	}
}
