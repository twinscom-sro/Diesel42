package environment;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;

public class Utilities {

    public static void writeFile(String outFile, StringBuilder sb) {
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(outFile));
            writer.write( sb.toString() );
            //writer.write("\n");
            writer.close();
            //System.out.println("Utilities::Writing file " + outFile);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static List<String> getFileNames(String folderPath) {
        List<String> fileNames = new ArrayList<String>();

        File directory = new File(folderPath);

        // Check if the path exists and is a directory
        if (directory.exists() && directory.isDirectory()) {
            File[] files = directory.listFiles(); // Get an array of File objects

            if (files != null) {
                for (File file : files) {
                    if (file.isFile()) { // Check if it's a file (not a subdirectory)
                        fileNames.add(folderPath+file.getName());
                        //System.out.println(file.getName()); // Print the file name
                    }
                }
            } else {
                System.out.println("Directory is empty or an I/O error occurred.");
            }
        } else {
            System.out.println("The provided path is not a valid directory or does not exist.");
        }

        return fileNames;
    }

    public static String getTime() {
        Calendar cal = Calendar.getInstance();
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy/MM/dd HH:mm:ss");
        return sdf.format(cal.getTime());

    }

    public static String getTimeTag() {
        Calendar cal = Calendar.getInstance();
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss");
        return sdf.format(cal.getTime());

    }

}
