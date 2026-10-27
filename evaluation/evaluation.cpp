
#include "evaluation.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <functional>
#include <utility>

using namespace cv;
using namespace std;

vector<Point2f> loadGroundTruth(
    const string& csvPath, const string& imageName, vector<int>* ballIds)
{
    vector<Point2f> points;
    if (ballIds)
        ballIds->clear();
    ifstream file(csvPath);

    if (!file.is_open())
    {
        cerr << "Could not open: " << csvPath << endl;
        return points;
    }

    string line;

    while (getline(file, line))
    {
        stringstream ss(line);
        string filename;

        getline(ss, filename, ',');

        if (filename != imageName)
            continue;

        vector<string> values;
        string value;

        while (getline(ss, value, ','))
        {
            values.push_back(value);
        }

        // Each ball has a Row and Column
        for (size_t i = 0; i + 1 < values.size(); i += 2)
        {
            if (values[i].empty() || values[i + 1].empty())
                continue;

            float row = stof(values[i]);
            float col = stof(values[i + 1]);

            points.push_back(Point2f(col, row));
            // Preserve the CSV ball number even when earlier pairs are blank.
            if (ballIds)
                ballIds->push_back(static_cast<int>(i / 2 + 1));
        }

        break;
    }

    return points;
}

EvaluationResult evaluateDetections(const vector<Point2f>& predictions, const vector<Point2f>& groundTruth, double tolerance)
{
    // List valid matches, trying nearby ground-truth balls first.
    vector<vector<pair<double, size_t>>> candidates(predictions.size());

    for (size_t p = 0; p < predictions.size(); ++p)
    {
        for (size_t g = 0; g < groundTruth.size(); ++g)
        {
            double distance = hypot(
                double(predictions[p].x) - groundTruth[g].x,
                double(predictions[p].y) - groundTruth[g].y
            );
            if (distance <= tolerance)
                candidates[p].emplace_back(distance, g);
        }
        sort(candidates[p].begin(), candidates[p].end());
    }

    // Reassign earlier matches when needed to maximize the number of
    // one-to-one matches. Each ground-truth ball counts at most once.
    vector<int> matchedPrediction(groundTruth.size(), -1);
    function<bool(size_t, vector<bool>&)> tryMatch =
        [&](size_t p, vector<bool>& visited) -> bool
    {
        for (const auto& candidate : candidates[p])
        {
            size_t g = candidate.second;
            if (visited[g])
                continue;
            visited[g] = true;

            if (matchedPrediction[g] == -1 ||
                tryMatch(static_cast<size_t>(matchedPrediction[g]), visited))
            {
                matchedPrediction[g] = static_cast<int>(p);
                return true;
            }
        }
        return false;
    };

    EvaluationResult result;
    for (size_t p = 0; p < predictions.size(); ++p)
    {
        vector<bool> visited(groundTruth.size(), false);
        if (tryMatch(p, visited))
            ++result.TP;
    }

    // Store the final one-to-one matching.
    result.predictionToGroundTruth.assign(
        predictions.size(), -1
    );

    result.groundTruthToPrediction.assign(
        groundTruth.size(), -1
    );

    for (size_t g = 0; g < groundTruth.size(); ++g)
    {
        int p = matchedPrediction[g];

        if (p == -1)
            continue;

        result.predictionToGroundTruth[p] = static_cast<int>(g);
        result.groundTruthToPrediction[g] = p;
    }

    double totalError = 0;
    for (size_t g = 0; g < groundTruth.size(); ++g)
    {
        if (matchedPrediction[g] == -1)
            continue;
        const Point2f& prediction = predictions[matchedPrediction[g]];
        totalError += hypot(
            double(prediction.x) - groundTruth[g].x,
            double(prediction.y) - groundTruth[g].y
        );
    }

    result.FP = static_cast<int>(predictions.size()) - result.TP;
    result.FN = static_cast<int>(groundTruth.size()) - result.TP;
    if (!predictions.empty())
        result.precision = double(result.TP) / predictions.size();
    if (!groundTruth.empty())
        result.recall = double(result.TP) / groundTruth.size();
    if (result.precision + result.recall > 0)
        result.f1 = 2 * result.precision * result.recall /
                    (result.precision + result.recall);
    if (result.TP > 0)
        result.meanError = totalError / result.TP;

    return result;
}

vector<LabelledBall> loadLabelledGroundTruth(const string& csvPath,const string& imageName) {
    vector<LabelledBall> balls;
    ifstream file(csvPath);

    if (!file.is_open())
        return balls;

    string line;

    while (getline(file, line)) {
        stringstream ss(line);
        string filename;

        getline(ss, filename, ',');

        if (filename != imageName)
            continue;

        vector<string> values;
        string value;

        while (getline(ss, value, ','))
            values.push_back(value);

        // 15 numbered balls + white cue ball
        for (int i = 0; i < 16; ++i) {
            int index = 2 * i;

            if (index + 1 >= static_cast<int>(values.size()))
                break;

            if (values[index].empty() ||
                values[index + 1].empty())
                continue;

            float row = stof(values[index]);
            float col = stof(values[index + 1]);

            LabelledBall ball;
            ball.number = (i == 15) ? 0 : i + 1;
            ball.centre = Point2f(col, row);

            balls.push_back(ball);
        }

        break;
    }

    return balls;
}
