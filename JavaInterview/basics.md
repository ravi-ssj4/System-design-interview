### Strings
```java
// Multiple ways of declaring strings in Java
// HelloWorld.java
class HelloWorld {
    public static void main(String[] args) {
        // 1. Simplest way
        String name = "Ravi Ranjan";
        System.out.println(name);

        // 2. using the new keyword
        String name = new String("Ravi Ranjan");
        System.out.println(name);

        // Slight difference between method 1 and 2
        
        // Method 1: JVM Creates a memory block to store "Ravi Ranjan" pushes it to the "String pool" -> if another string is initialized with the same value -> assigns that to this one only -> does not create a new one -> preferred as it optimizes memory!
        
        // Method 2: JVM creates a new String object regardless if its the same value or not

        // This can be shown by this example:
        String literalString1 = "abc";
        String literalString2 = "abc";

        String objectString1 = new String("xyz");
        String objectString2 = new String("xyz");

        System.out.println(literalString1 == literalString2)
        // -> true
        System.out.println(objectString1 == objectString2)
        // -> false

        // String formatting:
        String name = "Ravi Ranjan"; // %s
        String country = "India"; 
        int age = 33; // %d
        String company = "Anomaly";
        double gpa = 4.9; // %f
        char percentSign = '%'; // %c
        boolean amITellingTheTruth = false; // %b

        // System.out.println("My name is " + name + ". I am from " + country + ". I am " + age + "years old. I work for " + company + ". My GPA is " + gpa + ".")

        String formattedString = String.format("My name is %s. I am from %s. I am %d years old. I work for %s company. My GPA is %f. I have attended 100%c of my classes. All these are %b claims.", name, country, age, company, gpa, percentSign, amITellingTheTruth);
        System.out.println(formattedString);

        // String methods:
        String name = "Ravi Ranjan";
        System.out.println(name.length()); // 11 
        System.out.println(name.isEmpty()); // false 
        System.out.println(name.toUpperCase()); // RAVI RANJAN
        System.out.println(name); // Ravi Ranjan (original string is unchanged)
        System.out.println(name.toLowerCase()); // ravi ranjan

        // Object string comparison
        String string1 = new String("abc");
        String string2 = new String("abc");

        System.out.println(string1 == string2); // false
        System.out.println(string1.equals(string2)) // true 

        // String replacement
        string string = "The sky is blue.";
        System.out.println(string.replace("blue", "golden"));
        // The sky is golden.
        System.out.println(string); // The sky is blue. -> original string unchanged
        String updatedString = string.replace("blue", "golden");
        System.out.println(updatedString); // The sky is golden; -> the updated string is returned back and assigned to the updatedString var.

        // contains method
        System.out.println(string.contains("sky")); // true

        // VVI : User inputs
        // System.in -> standard input ( from terminal / console )
        // To take input in Java, we must use Scanner class object
        Scanner scanner = new Scanner(System.in); 
        System.out.println("Enter your name: "); 
        // System.out.print("Enter your name: \n); // same as above
        String name = scanner.nextLine();
        System.out.println(name); 
    }
}

```