function result = epsilon(Theta, epsilon)
    % result = ((Theta + epsilon) ^ 3 - (Theta - epsilon) ^ 3) ...
    %     / (2 * epsilon);


     result = ( ((3 * (Theta + epsilon) ^ 4) + 1) - ((3 * (Theta - epsilon) ^ 4) + 1) ) ...
        / (2 * epsilon);

