## Q. Why to use Integer instead of int and when to use it?
## A. The choice between using `Integer` and `int` in Java depends on the specific requirements and considerations of the application. Here are some reasons why you might choose `Integer` over `int`:

### 1. **Nullability**
- **`Integer`**: Can represent `null`. This is useful in situations where you need to distinguish between a value that is explicitly set to zero and a value that is uninitialized or missing.
- **`int`**: Primitive type that cannot represent `null`. It always has a default value of `0`.

### 2. **Object-Oriented Features**
- **`Integer`**: Being an object, `Integer` can be used in collections that require objects, such as `List<Integer>`, `Map<Integer, ...>`, etc. It also allows for the use of methods and features provided by the `Integer` class.
- **`int`**: Cannot be used in collections directly (requires boxing to `Integer`).

### 3. **Framework and Library Compatibility**
- Many Java frameworks (like Hibernate, Spring, etc.) and libraries expect the use of wrapper classes (`Integer`, `Long`, etc.) instead of primitives because they need to work with objects and handle `null` values.

### 4. **Database Integration**
- When working with ORM frameworks like Hibernate, fields that map to database columns often use wrapper classes (`Integer` instead of `int`) because databases can have nullable fields, and the ORM framework needs to handle `null` values correctly.

### Example Use Case: Hibernate Entity

In the context of an entity class that maps to a database table, using `Integer` for a field like `quantity` allows the field to be `null`, reflecting the state of the database column if it is nullable.

### Code Example

Here is your class with explanations for each field:

```java
import java.math.BigDecimal;

public class OrderLineItems {

    private Long id; // Using Long for possible nullability and to work with frameworks
    private String skuNumber; // String is inherently nullable
    private BigDecimal price; // BigDecimal is used for precise financial calculations and is an object
    private Integer quantity; // Using Integer to allow for null values, useful in database mapping and frameworks

    // Getters and Setters
    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getSkuNumber() {
        return skuNumber;
    }

    public void setSkuNumber(String skuNumber) {
        this.skuNumber = skuNumber;
    }

    public BigDecimal getPrice() {
        return price;
    }

    public void setPrice(BigDecimal price) {
        this.price = price;
    }

    public Integer getQuantity() {
        return quantity;
    }

    public void setQuantity(Integer quantity) {
        this.quantity = quantity;
    }
}
```

### Summary
Using `Integer` instead of `int` offers greater flexibility, especially in scenarios where nullability is important or when working with frameworks and libraries that require objects. This decision is often driven by the need to handle nullable values gracefully and to integrate smoothly with Java's object-oriented features and third-party tools.