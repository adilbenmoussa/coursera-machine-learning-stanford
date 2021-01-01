function plotDataAndCentroids(X, centroids, initial_centroids)

plot(X(:, 1), X(:, 2), 'k+','LineWidth', 1, 'MarkerSize', 7);
hold on;
% plot(X(1, 1), X(1, 2), 'kd','MarkerFaceColor', 'r', 'MarkerSize', 7);
% hold on;
% plot(X(2, 1), X(2, 2), 'kd','MarkerFaceColor', 'b', 'MarkerSize', 7);
% hold on;
% plot(X(290, 1), X(290, 2), 'kd','MarkerFaceColor', 'r', 'MarkerSize', 7);
% hold on;
plot(initial_centroids(:, 1), initial_centroids(:, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
plot(centroids(:, 1), centroids(:, 2), 'ko', 'MarkerFaceColor', 'b', 'MarkerSize', 7);
hold off;