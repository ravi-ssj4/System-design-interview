from abc import ABC, abstractmethod
import threading
from typing import List, Optional, Dict, Deque
from collections import deque
import heapq

class Elevator:
    def __init__(self, id: int):
        self.id: int = id
        self.current_floor: int = 0
        self.internal_requests: Deque[int] = deque()
        self.external_requests: Deque[int] = deque()
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
        
        elif self.external_requests and (not self.internal_requests or 
                self.external_requests[0] < self.internal_requests[0]):
            
            self.direction = "UP" if self.external_requests[0] > self.current_floor else "DOWN"
        
        elif self.internal_requests:
            
            self.direction = "UP" if self.internal_requests[0] > self.current_floor else "DOWN"

    def move(self) -> None:
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



class DispatchStrategy(ABC):
    @abstractmethod
    def assign_requests(self, elevators: List['Elevator'], request_queue: Deque['Request']) -> None:
        pass

class FCFSStrategy(DispatchStrategy):
    # at a time, it pops the latest request from the front of the queue and tries to find the best elevator for it
    # if it finds a best elevator, it adds this request to that elevator's individual request queue
    def assign_requests(self, elevators: List['Elevator'], request_queue: Deque['Request']) -> None:
        if request_queue:
            request = request_queue.popleft()
            best_elevator = self.find_best_elevator(elevators, request)
            if best_elevator:
                best_elevator.add_external_request(request.floor)
            else:
                request_queue.append(request)  # Re-queue the request if no suitable elevator found
        else:
            print("No requests in the Queue currently")

    # iterates through each elevator and calculates minimum distance of that elevator from the request
    # (ie. the person)
    # for the most suitable elevator, which is closest + in the same direction + hasn't passed the passenger
    # that elevator is returned as the best choice
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
                    elevator.add_external_request(request.floor)
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
    
class Request:
    def __init__(self, floor: int, direction: str):
        self.floor: int = floor
        self.direction: str = direction  # "UP" or "DOWN"
    

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