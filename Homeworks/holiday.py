flights = {1: 'Luxembourg', 2: 'Bangladesh', 3:'Copenhagen', 4:'Manchester'}

print('Where do you want to fly?')
for key in flights:
    print(key, '-', flights[key])

city_flight = int(input('Enter number: '))
num_nights = int(input('How long you want to stay there? '))
rental_days = int(input('For how many days you want to rent a car there? '))

def hotel_cost(num_nights):
    return num_nights*100

def plane_cost(city_flight):
    # Hint: use if/else if statements in the function to retrieve a price based on the chosen city.
    ticket_prices = {1: 200, 2:1000, 3:300, 4:80}  # Sorry I am not a fan of WET coding
    return ticket_prices[city_flight]

def car_rental(rental_days):
    return rental_days*40

def holiday_cost(city_flight, num_nights, rental_days):
    flight = plane_cost(city_flight)
    hotel = hotel_cost(num_nights)
    car = car_rental(rental_days)
    
    print('\n\nCost of your holiday:\n\nFlight to',flights[city_flight], flight, '\nPrice for hotel     ', hotel, '\nPrice for car rental', car, '\n\nTotal cost for trip:', flight+hotel+car)
    if rental_days > num_nights:
        print('Car rental company will love you.')

holiday_cost(city_flight, num_nights, rental_days)
