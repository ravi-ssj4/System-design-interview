Sure, let's revise the solution to use a priority queue for SSTF and hashmaps for SCAN and LOOK.

### 1. Strategy Interfaces

First, we define an interface for the dispatch strategies.

```python
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Deque
from collections import deque
import heapq

class DispatchStrategy(ABC):
    @abstractmethod
    def assign_requests(self, elevators: List['Elevator'], request_queue: Deque['Request']) -> None:
        pass
```

### 2. FCFS Strategy

The FCFS strategy will remain unchanged since it already uses a queue.

```python
class FCFSStrategy(DispatchStrategy):
    def assign_requests(self, elevators: List['Elevator'], request_queue: Deque['Request']) -> None:
        for _ in range(len(request_queue)):
            request = request_queue.popleft()
            best_elevator = self.find_best_elevator(elevators, request)
            if best_elevator:
                best_elevator.add_request(request.floor)
            else:
                request_queue.append(request)  # Re-queue the request if no suitable elevator found

    def find_best_elevator(self, elevators: List['Elevator'], request: 'Request') -> Optional['Elevator']:
        best_choice: Optional['Elevator'] = None
        min_distance: float = float('inf')
        
        for elevator in elevators:
            distance: float
            if elevator.direction == "IDLE":
                distance = abs(elevator.current_floor - request.floor)
            elif elevator.direction == request.direction:
                if request.direction == "UP" and request.floor >= elevator.current_floor:
                    distance = request.floor - elevator.current_floor
                elif request.direction == "DOWN" and request.floor <= elevator.current_floor:
                    distance = elevator.current_floor - request.floor
                else:
                    continue
            else:
                continue

            if distance < min_distance:
                min_distance = distance
                best_choice = elevator

        return best_choice
```

### 3. SSTF Strategy with Priority Queue

```python
class SSTFStrategy(DispatchStrategy):
    def assign_requests(self, elevators: List['Elevator'], request_queue: Deque['Request']) -> None:
        for elevator in elevators:
            if elevator.direction == "IDLE" and request_queue:
                priority_queue = []
                for request in request_queue:
                    distance = abs(elevator.current_floor - request.floor)
                    heapq.heappush(priority_queue, (distance, request))
                
                while priority_queue:
                    _, request = heapq.heappop(priority_queue)
                    elevator.add_request(request.floor)
                    request_queue.remove(request)

    def find_best_elevator(self, elevators: List['Elevator'], request: 'Request') -> Optional['Elevator']:
        best_choice: Optional['Elevator'] = None
        min_distance: float = float('inf')
        
        for elevator in elevators:
            distance = abs(elevator.current_floor - request.floor)
            if distance < min_distance:
                min_distance = distance
                best_choice = elevator

        return best_choice
```

### 4. SCAN Strategy with HashMap

```python
class SCANStrategy(DispatchStrategy):
    def assign_requests(self, elevators: List['Elevator'], request_queue: Deque['Request']) -> None:
        floor_map: Dict[int, List[Request]] = {}
        for request in request_queue:
            if request.floor not in floor_map:
                floor_map[request.floor] = []
            floor_map[request.floor].append(request)

        for elevator in elevators:
            if elevator.direction == "IDLE" and floor_map:
                self.scan_elevator(elevator, floor_map, request_queue)
    
    def scan_elevator(self, elevator: 'Elevator', floor_map: Dict[int, List['Request']], request_queue: Deque['Request']) -> None:
        for floor in sorted(floor_map.keys()):
            if elevator.direction == "IDLE":
                if floor >= elevator.current_floor:
                    elevator.direction = "UP"
                else:
                    elevator.direction = "DOWN"
            
            if elevator.direction == "UP" and floor >= elevator.current_floor:
                for request in floor_map[floor]:
                    elevator.add_request(request.floor)
                    request_queue.remove(request)
                del floor_map[floor]
            elif elevator.direction == "DOWN" and floor <= elevator.current_floor:
                for request in floor_map[floor]:
                    elevator.add_request(request.floor)
                    request_queue.remove(request)
                del floor_map[floor]

    def find_best_elevator(self, elevators: List['Elevator'], request: 'Request') -> Optional['Elevator']:
        best_choice: Optional['Elevator'] = None
        min_distance: float = float('inf')
        
        for elevator in elevators:
            if elevator.direction == "IDLE" or (elevator.direction == "UP" and request.floor >= elevator.current_floor) or (elevator.direction == "DOWN" and request.floor <= elevator.current_floor):
                distance = abs(elevator.current_floor - request.floor)
                if distance < min_distance:
                    min_distance = distance
                    best_choice = elevator

        return best_choice
```

### 5. LOOK Strategy with HashMap

```python
class LOOKStrategy(DispatchStrategy):
    def assign_requests(self, elevators: List['Elevator'], request_queue: Deque['Request']) -> None:
        floor_map: Dict[int, List[Request]] = {}
        for request in request_queue:
            if request.floor not in floor_map:
                floor_map[request.floor] = []
            floor_map[request.floor].append(request)

        for elevator in elevators:
            if elevator.direction == "IDLE" and floor_map:
                self.look_elevator(elevator, floor_map, request_queue)
    
    def look_elevator(self, elevator: 'Elevator', floor_map: Dict[int, List['Request']], request_queue: Deque['Request']) -> None:
        floors = sorted(floor_map.keys())
        if elevator.direction == "IDLE":
            if any(floor >= elevator.current_floor for floor in floors):
                elevator.direction = "UP"
            else:
                elevator.direction = "DOWN"
        
        if elevator.direction == "UP":
            for floor in floors:
                if floor >= elevator.current_floor:
                    for request in floor_map[floor]:
                        elevator.add_request(request.floor)
                        request_queue.remove(request)
                    del floor_map[floor]
        elif elevator.direction == "DOWN":
            for floor in reversed(floors):
                if floor <= elevator.current_floor:
                    for request in floor_map[floor]:
                        elevator.add_request(request.floor)
                        request_queue.remove(request)
                    del floor_map[floor]

    def find_best_elevator(self, elevators: List['Elevator'], request: 'Request') -> Optional['Elevator']:
        best_choice: Optional['Elevator'] = None
        min_distance: float = float('inf')
        
        for elevator in elevators:
            if elevator.direction == "IDLE" or (elevator.direction == "UP" and request.floor >= elevator.current_floor) or (elevator.direction == "DOWN" and request.floor <= elevator.current_floor):
                distance = abs(elevator.current_floor - request.floor)
                if distance < min_distance:
                    min_distance = distance
                    best_choice = elevator

        return best_choice
```

### 6. Elevator Class

```python
class Elevator:
    def __init__(self, id: int):
        self.id: int = id
        self.current_floor: int = 0
        self.requests: Deque[int] = deque()
        self.direction: str = "IDLE"  # Can be "UP", "DOWN", or "IDLE"
    
    def add_request(self, floor: int) -> None:
        self.requests.append(floor)
        if self.direction == "IDLE":
            self.update_direction()

    def update_direction(self) -> None:
        if not self.requests:
            self.direction = "IDLE"
        elif self.requests[0] > self.current_floor:
            self.direction = "UP"
        elif self.requests[0] < self.current_floor:
            self.direction = "DOWN"

    def move(self) -> None:
        if self.direction == "UP":
            self.current_floor += 1
        elif self.direction == "DOWN":
            self.current_floor -= 1
        self.update_direction()

    def step(self) -> None:
        if self.direction != "IDLE":
            self.move()
            if self.requests and self.requests[0] == self.current_floor:
                self.requests.popleft()
                if not self.requests:
                    self.direction = "IDLE"
                else:
                    self.update_direction()

    def __str__(self) -> str:
        return f"Elevator {self.id}: Floor {self.current_floor}, Direction: {self.direction}, Requests: {list(self.requests)}"
```

### 7. Request Class

```python
class Request:
    def __init__(self, floor: int, direction: str):
        self.floor: int = floor
        self.direction: str = direction  # "UP" or "DOWN"
```

### 8. ElevatorController Class

```python
class ElevatorController:
    def __init__(self, num_elevators: int, strategy: DispatchStrategy):
        self.elevators: List[Elevator] = [Elevator(id) for id in range(num_elevators)]
        self.request_queue: Deque[Request] = deque()
        self.strategy: DispatchStrategy = strategy

    def request_elevator(self, floor: int, direction: str) -> None:
        request = Request(floor, direction)
        self.request_queue.append(request)

    def step(self) -> None:
        self.strategy.assign_requests(self.elevators, self.request_queue)
        for

 elevator in self.elevators:
            elevator.step()

    def status(self) -> None:
        for elevator in self.elevators:
            print(elevator)
```

### 9. Simulation

```python
if __name__ == "__main__":
    # Change strategy here
    strategy = FCFSStrategy()
    # strategy = SSTFStrategy()
    # strategy = SCANStrategy()
    # strategy = LOOKStrategy()
    
    controller = ElevatorController(num_elevators=3, strategy=strategy)

    # Simulate some requests
    controller.request_elevator(3, "UP")
    controller.request_elevator(5, "DOWN")
    controller.request_elevator(1, "UP")

    for _ in range(10):
        controller.step()
        controller.status()
        print("---")
```

### Explanation

1. **DispatchStrategy Interface**:
   - Abstract base class for all dispatch strategies.

2. **FCFSStrategy**:
   - Implements the FCFS dispatch strategy using a queue.

3. **SSTFStrategy**:
   - Implements the SSTF dispatch strategy using a priority queue.

4. **SCANStrategy**:
   - Implements the SCAN dispatch strategy using a hashmap.

5. **LOOKStrategy**:
   - Implements the LOOK dispatch strategy using a hashmap.

6. **Elevator Class**:
   - Manages the state and behavior of an individual elevator.
   - Handles movement and requests using a `deque`.

7. **Request Class**:
   - Represents an elevator request with a specific floor and direction.

8. **ElevatorController Class**:
   - Manages multiple elevators and handles requests.
   - Uses a strategy pattern to assign requests based on the selected dispatch strategy.
   - Steps through the simulation, moving elevators and assigning requests.

9. **Simulation**:
   - Initializes the controller with three elevators and a chosen strategy.
   - Makes several elevator requests.
   - Steps through the simulation, printing the status of each elevator.

This code provides a flexible system where the dispatch strategy can be easily changed by modifying the `strategy` instance in the simulation section.

## Concurrency + Internal Elevator requests handled
To handle passengers pressing destination floor buttons after entering the elevator, we need to incorporate internal requests into the elevator's state. Additionally, concurrency might be required for a real-world elevator system to handle multiple requests simultaneously and ensure smooth operation. 

Here’s the revised implementation incorporating these aspects:

### Revised Elevator Class

```python
from typing import List, Deque, Dict
from collections import deque
import threading

class Elevator:
    def __init__(self, id: int):
        self.id: int = id
        self.current_floor: int = 0
        self.external_requests: Deque[int] = deque()
        self.internal_requests: Deque[int] = deque()
        self.direction: str = "IDLE"  # Can be "UP", "DOWN", or "IDLE"
        self.lock = threading.Lock()
    
    def add_external_request(self, floor: int) -> None:
        with self.lock:
            self.external_requests.append(floor)
            if self.direction == "IDLE":
                self.update_direction()

    def add_internal_request(self, floor: int) -> None:
        with self.lock:
            self.internal_requests.append(floor)
            if self.direction == "IDLE":
                self.update_direction()

    def update_direction(self) -> None:
        if not self.external_requests and not self.internal_requests:
            self.direction = "IDLE"
        elif self.external_requests and (not self.internal_requests or self.external_requests[0] < self.internal_requests[0]):
            self.direction = "UP" if self.external_requests[0] > self.current_floor else "DOWN"
        elif self.internal_requests:
            self.direction = "UP" if self.internal_requests[0] > self.current_floor else "DOWN"

    def move(self) -> None:
        with self.lock:
            if self.direction == "UP":
                self.current_floor += 1
            elif self.direction == "DOWN":
                self.current_floor -= 1
            self.update_direction()

    def step(self) -> None:
        with self.lock:
            if self.direction != "IDLE":
                self.move()
                if self.external_requests and self.external_requests[0] == self.current_floor:
                    self.external_requests.popleft()
                    self.update_direction()
                if self.internal_requests and self.internal_requests[0] == self.current_floor:
                    self.internal_requests.popleft()
                    self.update_direction()

    def __str__(self) -> str:
        return f"Elevator {self.id}: Floor {self.current_floor}, Direction: {self.direction}, External Requests: {list(self.external_requests)}, Internal Requests: {list(self.internal_requests)}"
```

### Revised Request Class

```python
class Request:
    def __init__(self, floor: int, direction: str):
        self.floor: int = floor
        self.direction: str = direction  # "UP" or "DOWN"
```

### Revised ElevatorController Class

```python
class ElevatorController:
    def __init__(self, num_elevators: int, strategy: DispatchStrategy):
        self.elevators: List[Elevator] = [Elevator(id) for id in range(num_elevators)]
        self.request_queue: Deque[Request] = deque()
        self.strategy: DispatchStrategy = strategy

    def request_elevator(self, floor: int, direction: str) -> None:
        request = Request(floor, direction)
        self.request_queue.append(request)

    def internal_request(self, elevator_id: int, floor: int) -> None:
        for elevator in self.elevators:
            if elevator.id == elevator_id:
                elevator.add_internal_request(floor)
                break

    def step(self) -> None:
        self.strategy.assign_requests(self.elevators, self.request_queue)
        for elevator in self.elevators:
            elevator.step()

    def status(self) -> None:
        for elevator in self.elevators:
            print(elevator)
```

### Concurrency Handling

To handle concurrency, we use locks within the `Elevator` class to ensure that state changes are thread-safe. For a real-world application, you might need more sophisticated concurrency control, but for this example, threading and simple locks suffice.

### Simulation

```python
if __name__ == "__main__":
    # Change strategy here
    strategy = FCFSStrategy()
    # strategy = SSTFStrategy()
    # strategy = SCANStrategy()
    # strategy = LOOKStrategy()
    
    controller = ElevatorController(num_elevators=3, strategy=strategy)

    # Simulate some requests
    controller.request_elevator(3, "UP")
    controller.request_elevator(5, "DOWN")
    controller.request_elevator(1, "UP")

    # Simulate internal requests (e.g., passengers pressing destination floor buttons)
    controller.internal_request(0, 7)
    controller.internal_request(1, 2)
    controller.internal_request(2, 9)

    for _ in range(10):
        controller.step()
        controller.status()
        print("---")
```

### Explanation

1. **Elevator Class**:
   - Manages the state and behavior of an individual elevator.
   - Handles both external and internal requests using separate deques.
   - Uses a lock for thread-safe state changes.
   - Updates direction based on both internal and external requests.
   - Moves the elevator and processes requests.

2. **Request Class**:
   - Represents an elevator request with a specific floor and direction.

3. **ElevatorController Class**:
   - Manages multiple elevators and handles requests.
   - Uses a strategy pattern to assign requests based on the selected dispatch strategy.
   - Handles internal requests from passengers pressing destination floor buttons.
   - Steps through the simulation, moving elevators and assigning requests.

4. **Simulation**:
   - Initializes the controller with three elevators and a chosen strategy.
   - Makes several external elevator requests.
   - Simulates internal requests from passengers.
   - Steps through the simulation, printing the status of each elevator.

This revised implementation now includes handling of internal requests (passengers pressing destination floor buttons) and ensures thread-safe state changes using locks.