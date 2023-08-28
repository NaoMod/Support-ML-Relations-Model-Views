package org.atlanmod.slepaper.modeling.generators.helpers;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URL;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public class DownloadAndExtractCSV {

	public static String here = new File(".").getAbsolutePath();

	/**
	 * Method to execute the download and unzip of all Movie Lens files into a temporary (not committed) folder
	 */
	public static String execute() {
		String urlStr = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip";
		String downloadPath = here + "/movielens/data.zip";
		String extractFolder = here + "/movielens/";

		try {
			// Download the ZIP file
			downloadFile(urlStr, downloadPath);

			// Extract the ZIP file
			extractZip(downloadPath, extractFolder);

			System.out.println("MovieLens files are downloaded and extracted successfully.");
			
			return extractFolder;
		} catch (IOException e) {
			e.printStackTrace();
		}
		return extractFolder;
	}

	/**
	 * Method to downalod a file from a URL
	 * 
	 * @param urlStr
	 * @param outputPath
	 * @throws IOException
	 */
	public static void downloadFile(String urlStr, String outputPath) throws IOException {
		URL url = new URL(urlStr);
		try (BufferedInputStream in = new BufferedInputStream(url.openStream());
				BufferedOutputStream out = new BufferedOutputStream(new FileOutputStream(outputPath))) {
			byte[] buffer = new byte[1024];
			int bytesRead;
			while ((bytesRead = in.read(buffer, 0, buffer.length)) != -1) {
				out.write(buffer, 0, bytesRead);
			}
		}
	}

	/**
	 * Method to unzip a file
	 * 
	 * @param zipFilePath
	 * @param extractPath
	 * @throws IOException
	 */
	public static void extractZip(String zipFilePath, String extractPath) throws IOException {
		try (ZipInputStream zipInputStream = new ZipInputStream(
				new BufferedInputStream(new FileInputStream(zipFilePath)))) {
			ZipEntry entry;
			while ((entry = zipInputStream.getNextEntry()) != null) {
				String entryName = entry.getName();
				if (!entry.isDirectory()) {
					File entryFile = new File(extractPath, entryName);
					if (!entryFile.getParentFile().exists()) {
						entryFile.getParentFile().mkdirs();
					}
					try (FileOutputStream fos = new FileOutputStream(entryFile)) {
						byte[] buffer = new byte[1024];
						int bytesRead;
						while ((bytesRead = zipInputStream.read(buffer)) != -1) {
							fos.write(buffer, 0, bytesRead);
						}
					}
				} else {
					File dir = new File(extractPath, entryName);
					if (!dir.exists()) {
						dir.mkdirs();
					}
				}
				zipInputStream.closeEntry();
			}
		}
	}
}
