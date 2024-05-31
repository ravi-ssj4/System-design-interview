### Integration tests (testcontainers.org)

```xml
<!-- Copy the following dependencies for Maven(pom.xml) from testcontainers.org -->

<dependencyManagement>
		<dependencies>
			<dependency>
				<groupId>org.testcontainers</groupId>
				<artifactId>testcontainers-bom</artifactId>
				<version>1.19.8</version>
				<type>pom</type>
				<scope>import</scope>
			</dependency>
		</dependencies>
	</dependencyManagement>

<!-- Note: we do not include the version tag here as well as its 
already there in the dependencyManagement(ie. testcontainers-bom) -> this 
is how the bill of materials concept works -> we can use multiple test 
containers and not specify the version as  its already there in the main 
dependencyManagement -->
<dependency>
			<groupId>org.testcontainers</groupId>
			<artifactId>mongodb</artifactId>
<!--			<version>1.19.8</version>--> 
			<scope>test</scope>
		</dependency>
		<dependency>
			<groupId>org.testcontainers</groupId>
			<artifactId>junit-jupiter</artifactId>
<!--			<version>1.19.8</version>-->
			<scope>test</scope>
		</dependency>
```

```java
package com.codeverse.productservice;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.testcontainers.containers.MongoDBContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

@SpringBootTest
@Testcontainers
class ProductServiceApplicationTests {

	@Container
	static MongoDBContainer mongoDBContainer = new MongoDBContainer("mongo:4.4.2");

	@DynamicPropertySource
	static void setProperties(DynamicPropertyRegistry dynamicPropertyRegistry) {
		dynamicPropertyRegistry.add("spring.data.mongodb.uri", mongoDBContainer::getReplicaSetUrl);
	}

	@Test
	void contextLoads() {
	}

}

// now we are ready to write integration tests:

// 1. Post request test: input ProductRequest, output: http status = "created"

```

### Inter-process Communication between Services
```java
// Imagine: 

// Sync communication: Service A is sending request to Service B and waiting for its response from B
// Async communication: Service A sends request to Service B but dosen't care about response from B

// Synchronous communication between Order-service & Inventory-service:
// Eg. Customer places an order 
// -> Order-service needs to check if the product is in stock or not 
// -> communicates with the Inventory service

// Rest Template(spring boot framework) ( still in maintenence mode)
// Web Client(spring web flux project) ( hence, preferred )
 
```

### Service Discovery Pattern using Netflix Eureka
```java
// practical scenario -> multiple instances of inventory-service running

// order-service can connect to any instance of inventory-service

// We create a new module for service discovery : Eureka server (discovery-server)
// order-service, product-service, inventory-service all are Eureka clients
// All Eureka clients connect to Eureka Server
// // Eureka server has knowledge of all the instances of the clients connected to it(ie. inventory-service, etc)
// When the order-service sends a request -> asks Eureka server for the inventory-service's address ->
// Eureka server gives the addresses of all the instances of the inventory-service -> order-service saves it in its local for caching later
// -> order-service has client side load balancing implemented(config/WebClientConfig updated!) and hence, goes through its list of instances
// (which it received previously from discovery-server(Eureka server) and cached), and sends the request to the first available instance

// To test the fact that the order-service is keeping the inventory-service registry(given by Eureka server previously) in its local:
// Exp 1: Terminate the discovery-service -> we are still able to send the post request to create the order from postman -> reason is that the order-service
// still had the info about the instances of the inventory-service in its local
// Exp 2: Now, we terminate all instances of inventory-service and restart a single instance(will start on a different random port(dynamic)) -> order creation
// request fails this time! -> order-service does not have info about this new instance(running on a new port) in its local


```

### API Gateway using Spring Cloud's own Gateway service
```java
// The api gateway provides a plethora of features:
// https://spring.io/projects/spring-cloud-gateway

// Note: The API gateway connects with all services within via http and not https(as everything is in our internal n/w and protected)
// Only the API gateway is exposed to the outside world, hence it receives https requests
// The main job of API Gateway:
// Accept the client request and re-route those requests to specific service (let's say product service has multiple instances)
// It does so by providing load balancing out of the box

```