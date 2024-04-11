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
/*

    let and const are more strict than var

    var's memory:
        allocated during the memory phase
        its allocated in the "global" section
    
    let & const's memory:
        allocated but they are "NOT AVAILABLE" during memory phase
        also, they are allocated in the "script" section

Note: Anything let or const be it a variable or object(function, array, etc) is allocated memory in the script section and not global!
    
    Temporal Dead Zone:
        * before execution phase, all of let, var and const -> memory allocated -> hoisted!
        * but, "var = undefined" and can be accessed
        * but, "let, const = <value unavailable>" 
        *   -> cannot be accessed -> this phase is called "Temporal Dead Zone"
*/

console.log(x) // temporal dead zone
// -> Uncaught Reference error: cannot access 'x' before initialization
console.log(y)
// -> Uncaught Reference error: cannot access 'y' before initialization
console.log(z)
// -> undefined

let x = 10
const y = 10
var z = 10

console.log(x) // -> 10
console.log(y) // -> 10
console.log(z) // -> 10


/*
let, const: BLOCK Scoped
var: FUNCTION Scoped
*/

{
    let a = 10
    const b = 20
    var c = 30
    
    console.log(a) // 10
    console.log(b) // 20
    console.log(c) // 30
}

console.log(a) // Uncaught ReferenceError: a is not defined
console.log(b) // Uncaught ReferenceError: b is not defined
console.log(c) // 30


/*
    Lexical Scoping:
        A local scope can access something from Global scope
        But vice versa not true, ie. cannot access local stuff from global scope

        Caveat: If there is a var in Global scope and in local scope its re-initialized, then local scope overrides global scope variable
*/
// =======================================================

function print(a) {
    console.log(a)
}

let a = 10
print(a) // 10

// =======================================================

function print(a) {
    a = 100
    console.log(a)
}

let a = 10
print(a) // 100


```

### Functions - First Class Citizens

```js

// Normal function -> allocated in the global section
function add(a, b) {
    return a + b
}

console.log(add)
console.log(add(2, 3))

// Assigning function to a variable -> allocated in the script section
let sum = function (a, b) {
    // here in the local execution context of this function, it has
    // reference to the outside global context as well: 
    // Local -> "this: Window" object
    return a + b
}

// Allocated in global section since "var"
var sum = function(a, b) {
    return a + b
}

console.log(sum)
console.log(sum(1, 3))

/*
    HIGHER ORDER FUNCTIONS:
        a higher order function is a function
        that takes one or more functions as args
        or returns a function
    
*/

// Passing function as an argument

let sum = function(a, b) {
    return a + b
}

let diff = function(a, b) {
    return a - b
}

let operate = function(operateFunction, a, b) {
    return operateFunction(a, b)
}

console.log(operate(sum, 2, 3)) // 5
console.log(operate(diff, 2, 3)) // -1


// Arrow Functions - ES6

let sum = function(a, b) {
    return a + b
}
// is same as:

let sum = (a, b) => {
    return a + b
}
// is same as:
let sum = (a, b) => a + b // only valid for 1 line content or exp


// Functions returned from other functions

// CASE 1
function outer() {
    function inner() {
        console.log("zidane")
    }
    return inner
}

let returnedFuncVar = outer()

console.log(returnedFuncVar) // prints the entire definition of inner()
returnedFuncVar() // calls inner() -> prints "hello"

// CASE 2

let a = 10;

function outer() {
    a = 100;
    function inner() {
        console.log(a)
    }
    return inner
}

let returnedFuncVar = outer()

console.log(returnedFuncVar) // prints the entire definition of inner()
returnedFuncVar() // calls inner() -> prints 100 -> even though the value of a  is taken from the global scope as inner() has reference to global scope object, still a = 100 overwrites that and a becomes 100 at line 369


// CASE 3

let a = 10;

function outer() {
    a = 100;
    function inner() {
        console.log(a)
    }
    return inner
}

let returnedFuncVar = outer()
a = 20
console.log(returnedFuncVar) // prints the entire definition of inner()
returnedFuncVar() // calls inner() -> prints 20 -> inner() has reference to global object, hence a = 20 when changed in the global scope, that's the value inner() takes

/*
    CLOSURES:
        * The above is an example of Closure in Javascript -> ES6 thing
        * Function + Lexical scoping
*/

function outer() {
    let count = 0;
    function inner() {
        count++;
        console.log(count);
    };
};

let incrementCounter = outer()
/*
    Execution context of incrementCounter:
        apart from "script" and reference to "global" section,
        this has a "closure" scope as well
        
        Closure (outer) count = 0, 1, 2...
        
        What is a closure here?
            * A closure is a function having access to parent function's scope
            * it basically takes a snapshot of the parent function's scope after the parent function has returned
            * Here, incrementCounter is the closure having the inner function and having access to outer() ie. parent function's scope, ie. count variable, even after the parent function has returned.  

    
        Issue with closures:
            * if too many closures -> too much memory needed to store global references for each closure -> MEMORY LEAKS
*/
incrementCounter() // 1
incrementCounter() // 2
incrementCounter() // 3
incrementCounter() // 4
incrementCounter() // 5
// ==============================================================================

// A better example to understand closures:

// parentFunction returns an anonymous function
// the anonymous function takes a snapshot of the parent's scope when returned and assigned to the const variables: son and daughter
const parentFunction = (person) => {
    let coins = 3

    return () => {
        coins -= 1
        if (coins > 0)
            console.log(`${person} has ${coins} left`)
        else
            console.log(`${person} is out of coins!`)
    }
}

const son = parentFunction('Son')
const daughter = parentFunction('Daughter')

// 2 separate closures are created for each
// Demo:

son() // Son has 2 left
son() // Son has 1 left
son() // Son is out of coins!

daughter() // Daughter has 2 left
son() // Son is out of coins! -> still the son is out of coins
daughter() // Daughter has 1 left


Link: "https://www.youtube.com/shorts/FcrdHbrBVgA"

// The anonymous inner function when returned takes the scope of the parent function, ie. coins here
// that means the parentFunction can give both son and daughter 3 coins separately
// here, both son and daughter vars are closures!
```

### Callbacks, Promises and Async Await

```js
/* 
    Callbacks: 
        * A callback is a function that's passed to another async function and is called after an async operation is done.
        * It basically is a handler to handle the response of the async function/operation
    Main use:
        * Callback fn's main use is to be called immediately after the async operation
        * ie. after the async operation is done, then only we want to call it
        
    Event Queue: 
        As soon as an async operation is registered, its put in an "event queue"
    
    Event loop:
        * Event loop keeps on checking the event queue for async operations to be done
        * as soon as its done, it picks it up and puts it on the call stack, ie. program scope
*/

// setTimeout mimics a server sending data back after 5 seconds!
// Assume: 
//    fetchData() is an asynchronous function that has an asynchronous operation that fetches some data after 5 seconds
function fetchData(callback) {
    setTimeout(() => {
        let data = 'fetched data';
        callback(data, null);
    }, 5000);
}

function handleData(data, error) {
    if (error) {
        console.error(error);
    } else {
        console.log(data);
    }
};

console.log('start')
fetchData(handleData) // fetched data printed after 5 seconds

// here, handle data is the callback function and we want it to be called only after the data is fetched from the async operation, not before that!

// Q. if there are multiple async operations and callbacks, then can the handleData be called after more than 5 seconds?
// A. Yes, due to event loop and event queue concept

// Callback Hell or Pyramid of Doom

asyncOperation1(arg1, (result1) => {
    asyncOperation2(arg2, (result2) => {
        asyncOperation3(arg3, (result3) => {
            // And so on.. 
        })
    })
})

/*
    here, we have anonymous callback functions nested one inside another
    Issues:
        Readability
        Difficult error handling
        Nested asyncOperation functions calling the callbacks instead of us controlling that -> Inversion of Control
    
    Solution: Promises / Async - Await
*/

/* 
    Promises:
        An object that represents the eventual state of an asynchronous operation
    
    * 3 states of promises:
        Pending
        Fulfilled
        Rejected
*/

// Assume: this is an async function having an async operation that returns a promise object
function getData() {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            data = "fetched data from server"
            if (data is fine)
                resolve(data)
            else
                reject("server error: ", data)
        }, 5000)
    })    
}

getData()
    .then(result => {
        console.log(result)
    })
    .catch(error => {
        console.error(error)
    })

// resolve, reject are also callbacks which can be called when the async operation is done depending on if the operation was successful or not

// here, the callback is 
// -> result => { console.log(result) }
// its in our control when we want to call it and how

/*
    Async Await:
        the logic from line 563 to line 569 can be written in a synchronous style
    NOTE: getData will still be the same!
*/

function getData() {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            data = "fetched data from server"
            if (data is fine)
                resolve(data)
            else
                reject("server error: ", data)
        }, 5000)
    })    
}

async function handleData() {
    try {
        const result = await getData()
        console.log(result)
    } catch(error) {
        console.error(error)
    }   
}

// A better example
Link: "https://www.youtube.com/watch?v=PoRJizFvM7s&list=PLillGF-RfqbbnEGy3ROiLWk7JMCuSyQtX&index=11&ab_channel=TraversyMedia"

```

### Common Higher order functions
1. Map: output has same num of elements as the input
2. Filter: output has <= elements as input
3. Reduce: output is only 1 value(like sum or some kind of aggregation of all inputs)

```js

// 1. Map

const nums = [1, 2, 3, 4, 5]
console.log(nums)
const doubledNums = nums.map((num) => 2*num)
console.log(doubledNums) // [2, 4, 6, 8, 10]

// 2. filter

const nums = [1, 2, 3, 4, 5]
console.log(nums)
const filteredNums = nums.filter((num) => num % 2 === 0)
console.log(filteredNums) // [2, 4]

// 3. Reduce

const nums = [1, 2, 3, 4, 5]
console.log(nums)
const sum = nums.reduce((accumulator, num) => accumulator + num, 0)
console.log(sum) // 15

const product = nums.reduce((accumulator, num) => accumulator * num, 1)
console.log(product) // 120

Overall_link: "https://www.youtube.com/watch?v=e2fKYP_7B_Y&ab_channel=KeertiPurswani"

```

### Javascript DOM
```js

```


### Javascript Object Oriented Programming
```js

```

