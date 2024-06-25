
import java.util.ArrayList;
import java.util.List;


public class LearnArrayList {
    public static void main(String[] args) {
        // String[] studentNames = new String[30];
        
        // studentNames[0] = "Bicky";
        // // .... .
        // studentNames[29] = "Rocky";

        // // newStudent
        // studentNames[30] = "Rambo"; // error


        // ArrayList<String> studentNames = new ArrayList<>();

        // studentNames.add("Rocky");
        // studentNames.add("Rajesh");

        // // prev size = n
        // // expansion: n + n / 2 + 1 = 1.5 times increase upon filling

        List<Integer> list = new ArrayList();
        list.add(1);
        list.add(2);
        list.add(3);
        System.out.println(list);

        list.add(4);
        System.out.println(list);

        list.add(1, 50);
        System.out.println(list);

        List<Integer> newList = new ArrayList();

        newList.add(150);
        newList.add(160);

        // adding/appending the new list into the original list
        list.addAll(newList);

        System.out.println(list);

    }
}