function r()
    % If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11.

    Theta1 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
    Theta2 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
    Theta3 = rand(1,11) * (2 * INIT_EPSILON) - INIT_EPSILON;