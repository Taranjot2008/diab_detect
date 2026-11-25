
# Importing Julia Libraries
using CSV
using DataFrames
using Flux
using Zygote
using Plots
using Statistics

# Importing the data file from the given path
filepath = joinpath(@__DIR__, "data", "diabetes.csv")

# Reading the data file
dataFile = CSV.read(filepath, DataFrame; delim = ',')

# Renaming the columns of the datafile to lowercase
rename!(dataFile, lowercase.(string.(names(dataFile))))

# Creating an array of data for X from the columns of the datafile
x_data = [[row.age, row.bmi, row.glucose, row.bloodpressure] for row in eachrow(dataFile)]

# Storing the last column of data (outcome) as the Y value
# It stores that whether the patient has diabetes or not
# 1 if the patient has diabetes or 0 if they do not
y_data = (dataFile.outcome)

# Creating a feature vector array of the data
xs = vcat(x_data)
ys = vcat(y_data)

# Reshaping the vectors into matrix of the specific size.
X = Float32.(hcat(xs...)) # shape: 4 x N
Y = Float32.(reshape(ys, 1, :)) # shape: 1 x N


for i in 1:size(X, 1)
    X[i, :] = (X[i, :] .- mean(X[i, :])) ./ std(X[i, :])
end

# Creating the model for training
# Using a chain model of Dense layers
# Using the ReLu and sigmoid activation function

# Different architecture models poposed:
# model = Chain(Dense(4, 16, relu), Dense(16, 1, σ))
# model = Chain(Dense(4, 8, relu), Dense(8, 4, relu), Dense(4, 1, σ))
# model = Chain(Dense(4, 16, relu), Dense(16, 8, relu), Dense(8, 1, σ))
model = Chain(Dense(4, 8, relu), Dense(8, 1, σ))

# Calculating the Loss during the training the model
# We use the Binary Cross Entropy function for calculating the loss
L(m, X, Y) = Flux.Losses.binarycrossentropy(m(X), Y)

# Applying the Adam optimization
# Calculating the model's weights and biases
opt = Flux.setup(Flux.Adam(0.001), model)

# Storing the initial and last value for loss
first_loss = nothing
last_loss = nothing


for epoch in 1:5000
    # Accessing the global loss variables
    global first_loss
    global last_loss

    # Calculating the gradients using automated differentiation
    grads = Zygote.gradient(m -> L(m, X, Y), model)
    g = grads[1]
    # Repeating the process until convergence
    Flux.update!(opt, model, g)

    # Printing the first loss at the first calculated epoch
    if epoch == 1
        first_loss = L(model, X, Y)
        println("Assigned firstloss = ", first_loss)
    end

    # Calculating the last loss at the end of the epoch range
    last_loss = L(model, X, Y)
end

# Printing the size of the X array
println("X size: ", size(X))
# Printing the size of the Y array
println("Y size: ", size(Y))

# Printing the final first and last loss
println("First Loss: ", first_loss)
println("Last Loss: ", last_loss)


model(X[:, 1])
model(X[:, end])

# Creating the prediction vector by passing the X array into the model
preds = vec(model(X))

# Calculating the binary output of the probability statistics
binary_preds = preds .>= 0.5

# Calculating the accuracy of the final prediction
accuracy = sum(binary_preds .== (vec(Y) .== 1)) / length(Y)

# Printing the accuracy
println("Accuracy = ", accuracy)

println("Min prediction: ", minimum(preds))
println("Max prediction: ", maximum(preds))
println("Mean prediction: ", mean(preds))


######################################
# HeatMap Visualization
######################################

avg_age = mean(X[1, :])
avg_bp = mean(X[4, :])

x_min, x_max = minimum(X[3, :]), maximum(X[3, :])
y_min, y_max = minimum(X[2, :]), maximum(X[2, :])

x_range = range(x_min, x_max, length = 200)
y_range = range(y_min, y_max, length = 200)


Z = zeros(length(x_range), length(y_range))

for i in 1:length(x_range)
    for j in 1:length(y_range)
        
        point = [
            avg_age,
            y_range[j],
            x_range[i],
            avg_bp
        ]

        Z[i, j] = model(point)[1]
    end
end

no_diabetes = vec(Y) .== 0
diabetes = vec(Y) .== 1

# plotting a heatmap
heatmap(
    x_range,
    y_range,
    Z',
    xlabel = "Glucose",
    ylabel = "BMI",
    color =:viridis
)

# scatter the points
scatter!(
    X[3, no_diabetes],
    X[2, no_diabetes],
    markersize = 3,
    color =:blue,
    label = "No Diabetes"
)

scatter!(
    X[3, diabetes],
    X[2, diabetes],
    markersize = 3,
    color =:red,
    label = "Diabetes"
)
