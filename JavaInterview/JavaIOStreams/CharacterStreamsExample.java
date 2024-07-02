import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class CharacterStreamsExample {
    public static void main(String[] args) throws IOException {
        FileReader reader = null;
        FileWriter writer = null;
        
        try {
            reader = new FileReader("D:\\Ravi\\2023\\Interview\\System-design-interview\\JavaInterview\\Java IO Streams\\source.txt");
            writer = new FileWriter("D:\\Ravi\\2023\\Interview\\System-design-interview\\JavaInterview\\Java IO Streams\\destination.txt");
            int content;
            while ((content = reader.read()) != -1) {
                writer.append((char) content);
            }
        }
        finally {
            if (reader != null)
                reader.close();
            if (writer != null)
                writer.close();
        }
    }
}