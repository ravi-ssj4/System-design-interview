### Intro
Q. why was javascript originally developed?
<br>A. Execution within web browsers only!

Q. Where does JS code run? <br> 
* JS code can be run on only JS execution engine
* Eg. Firefox-> spider monkey, chrome-> V8

Q. Can JS code run outside the browsers?
<br>A. Yes, Node.js was introduced for that purpose

### Node.js
* its a runtime environment built on top of V8 JS engine
* its written in C++
* Once, installed in the local machine, it can run JS code
* to run any JS program -> node prog_name.js 

### Faster way to run Javascript for testing:
* Create an html and load the javascript script into it
* Open the html with live server extension in vscode
* Go to browser and go to Developer Tools panel(F12) -> Console & Sources(for debugging JS Code)

### Variables
```js
// Primitive or Value types -> string, number, boolean, undefined, null

// Javascript is dynamically typed! -> line 50

// String
let x = "football"
console.log(x, typeof(x))

// number
let y = 512
console.log(y, typeof(y))

let a = 1.4
console.log(a, typeof(a)) // also a number type only (no floats / doubles)

// boolean
let z = true
console.log(z, typeof(z))

// undefined
let b;
console.log(b, typeof(b)) // type of b is undefined

b = "ravi"
console.log(b, typeof(b)) // type dynamically changed to string

b = undefined
// we forcefully made b == undefined > can be done > not recommended
console.log(b) // type = undefined -> not recommended

// what if I want the value to be emtpy?
b = null
console.log(b, typeof(b)) // type = object

``` 

### Reference types (Objects)
```js
// Reference types -> Objects, Arrays, Functions
// -> grouping of multiple things together

let sport = {
    name: "Football",
    description: "Mother of all games",
    rating: "infinite"
}

console.log(sport)
// -> the actual sport object

console.log(typeof(sport))
// -> Object type

// How to access
// 2 ways:
console.log(sport.name) // dot notation
// -> Football

console.log(sport['description']) // bracket notation
// -> Mother of all games

```

### Interview Question: Why are value types and reference types named so?
```js
let x = "messi"
let y = x

x = "ronaldo"

console.log(x) 
// -> ronaldo
console.log(y) 
// -> messi (y dosen't change as only the value of x was copied to y in line 91)

let p = {
    name: "messi"
}

let q = p

p.name = "ronaldo"

console.log(p)
// p = {
//     name: "ronaldo"
// }
console.log(q)
// q = {
//     name: "ronaldo"
// }

```
### Arrays
```js
// Just like Python lists - dynamic and can have multiple types inside them

let messi = ['golden boot', 34, ['tiago', 'mateo'], true]

console.log(messi[0]) -> 'golden boot'
console.log(messi[1]) -> 34
console.log(messi[2][1]) -> mateo

console.log(typeof(messi)) -> Object

```

### Interview Question: How Javascript code actually executes?
```js
/*

Execution Context:
    * Place constituting the vars, functions, scope chain, etc -> basically everything

Global Execution Context:
    * Upon running JS code, a Global Execution Context is created!

Local Execution context:
    * Everytime a function is called in a javascript code -> a new Local execution context is created inside the Global Execution context

2 phases of Execution contexts:
    1. Memory phase (variable environment): 
        * no line is executed
        * only memory is allocated to everything (vars and objects)
    2. Code phase (thread of execution):
        * the code is executed line by line 'synchronously' in a single thread

Note: Javascript is a Synchronous Single Threaded Language!

*/

// Demonstration:

// Memory phase already done before we even start
// playGame and x already allocated memory
// Now from line 161, code phase begins and continues till line 174
playGame('Hockey')
// -> playing Hockey -> works fine as playGame was already allocated and callable

console.log(x)
// -> undefined -> since x = 10 is done later, at this line 164, x = undefined

function playGame(gameName) {
    console.log('playing' + gameName)
}

var x = 10
console.log(x)
// -> 10
playGame('football')
// -> playing football

/* 
    HOISTING
        * The above concept of accessing or using vars, functions, etc. before initializing them
    
*/

// WINDOW === THIS === Global Object created when code execution starts!
console.log(a) // undefined
var a = 10
console.log(a) // 10
console.log(this.a) // 10
console.log(window.a) // 10
console.log(window) // entire global object printed
console.log(this === window) // true
```

### let, var and const
```js
    
```