To set up a JSON document and play with JsonPath queries in Visual Studio Code (VS Code), you can use the "REST Client" extension, which allows you to send HTTP requests and view the results directly in VS Code. Additionally, you can use an online tool like [JSONPath Online Evaluator](http://jsonpath.com/) to test your JsonPath expressions.

### Step-by-Step Setup in VS Code

1. **Install VS Code**:
   - If you haven't already, download and install Visual Studio Code from [here](https://code.visualstudio.com/).

2. **Install REST Client Extension**:
   - Open VS Code.
   - Go to the Extensions view by clicking on the Extensions icon in the Activity Bar on the side of the window or pressing `Ctrl+Shift+X`.
   - Search for "REST Client" and install it.

3. **Create a JSON File**:
   - Create a new file in VS Code and save it with a `.json` extension, for example, `data.json`.
   - Paste your JSON document into this file. Here's an example JSON document you can use:
     ```json
     {
       "store": {
         "book": [
           { "category": "reference", "author": "Nigel Rees", "title": "Sayings of the Century", "price": 8.95 },
           { "category": "fiction", "author": "Evelyn Waugh", "title": "Sword of Honour", "price": 12.99 },
           { "category": "fiction", "author": "Herman Melville", "title": "Moby Dick", "isbn": "0-553-21311-3", "price": 8.99 },
           { "category": "fiction", "author": "J.R.R. Tolkien", "title": "The Lord of the Rings", "isbn": "0-395-19395-8", "price": 22.99 }
         ],
         "bicycle": {
           "color": "red",
           "price": 19.95
         }
       }
     }
     ```

4. **Create a REST Client Request File**:
   - Create a new file and save it with a `.http` or `.rest` extension, for example, `requests.http`.
   - Add the following content to the file to create a POST request that sends the JSON document and a placeholder for JsonPath queries:
     ```http
     POST https://jsonpath.herokuapp.com/api HTTP/1.1
     Content-Type: application/json

     {
       "json": {
         "store": {
           "book": [
             { "category": "reference", "author": "Nigel Rees", "title": "Sayings of the Century", "price": 8.95 },
             { "category": "fiction", "author": "Evelyn Waugh", "title": "Sword of Honour", "price": 12.99 },
             { "category": "fiction", "author": "Herman Melville", "title": "Moby Dick", "isbn": "0-553-21311-3", "price": 8.99 },
             { "category": "fiction", "author": "J.R.R. Tolkien", "title": "The Lord of the Rings", "isbn": "0-395-19395-8", "price": 22.99 }
           ],
           "bicycle": {
             "color": "red",
             "price": 19.95
           }
         }
       },
       "path": "$.store.book[*].author"
     }
     ```

5. **Run the REST Request**:
   - Place your cursor on the `POST` line and click the "Send Request" button that appears above the line.
   - The REST Client will send the request, and the response will appear in a new tab.

### Using an Online Tool

For quick JsonPath evaluations, you can also use the [JSONPath Online Evaluator](http://jsonpath.com/):

1. **Open the JSONPath Online Evaluator**:
   - Go to [http://jsonpath.com/](http://jsonpath.com/).

2. **Paste Your JSON Document**:
   - Paste your JSON document into the left-hand side panel.

3. **Enter JsonPath Expressions**:
   - Enter your JsonPath expressions in the input box and see the results immediately.

### Example Queries

Here are some example JsonPath queries you can try:

1. **Get all authors**:
   ```jsonpath
   $.store.book[*].author
   ```

2. **Get the price of the first book**:
   ```jsonpath
   $.store.book[0].price
   ```

3. **Get all books with a price less than 10**:
   ```jsonpath
   $.store.book[?(@.price < 10)]
   ```

4. **Get all categories**:
   ```jsonpath
   $.store.book[*].category
   ```

5. **Get the color of the bicycle**:
   ```jsonpath
   $.store.bicycle.color
   ```

By following these steps, you can easily play around with JsonPath expressions in VS Code and see the results in real-time. This setup will help you get a better understanding of how JsonPath works.