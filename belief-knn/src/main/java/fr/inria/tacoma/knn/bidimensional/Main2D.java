package fr.inria.tacoma.knn.bidimensional;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.MappingIterator;
import com.fasterxml.jackson.databind.ObjectMapper;
import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.knn.LabelledPoint;
import fr.inria.tacoma.knn.generic.KnnBelief;
import fr.inria.tacoma.knn.unidimensional.SensorValue;
import fr.inria.tacoma.knn.util.KnnUtils;
import org.jfree.chart.ChartPanel;

import javax.swing.*;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class Main2D {

    public static final double ALPHA = 0.3;

    public static void main(String[] args) throws IOException {
        FrameOfDiscernment frame = FrameOfDiscernment.newFrame("presence", "presence", "absence");
        List<SensorValue> absence = getPoints("absence", "absence-motion1.json");
        List<SensorValue> presence = getPoints("presence", "presence-motion1.json");

        // transforming to 2 dimensions (lists must be ordered by timestamp)
        List<LabelledPoint<Coordinate>> absence2D = to2D(absence);
        List<LabelledPoint<Coordinate>> presence2D = to2D(presence);

        // extracting cross validation data
        List<LabelledPoint<Coordinate>> crossValidationPoints = KnnUtils.extractSubList(absence2D, 0.6);
        crossValidationPoints.addAll(KnnUtils.extractSubList(presence2D, 0.6));

        List<LabelledPoint<Coordinate>> trainingSet = new ArrayList<>();
        trainingSet.addAll(presence2D);
        trainingSet.addAll(absence2D);

        showBestMatchWithFixedAlpha(frame.toStateSet("presence"), frame, trainingSet,
                crossValidationPoints);
    }



    /**
     * Shows the model having the lowest error depending on K.
     *  @param frame       frame of discernment
     * @param points training set on which we apply knn.
     */
    private static void showBestMatchWithFixedAlpha(StateSet stateSet, FrameOfDiscernment frame,
                                                    List<LabelledPoint<Coordinate>> points,
                                                    List<LabelledPoint<Coordinate>> testSample) {
        show(stateSet, KnnUtils.getBestKnnBelief(frame, points, testSample, ALPHA,
                Coordinate::distance));

    }

    private static void show(StateSet stateSet, KnnBelief<Coordinate> model) {
        ChartPanel chartPanel = JfreeChartDisplay2D.getChartPanel(model, 250, 2000.0,
                250, 2000.0, stateSet);
        JFrame windowFrame = new JFrame();
        windowFrame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        windowFrame.setContentPane(chartPanel);
        windowFrame.pack();
        windowFrame.setVisible(true);
    }



    private static List<LabelledPoint<Coordinate>> to2D(List<SensorValue> sortedPoints) {
        List<LabelledPoint<Coordinate>> points = new ArrayList<>(sortedPoints.size() - 1);
        for (int i = 1; i < sortedPoints.size(); i++) {
            SensorValue point = sortedPoints.get(i);
            SensorValue prevPoint = sortedPoints.get(i - 1);
            double absoluteDerivative = Math.abs(point.getValue() - prevPoint.getValue())
                    / (point.getTimestamp() - prevPoint.getTimestamp());
            Coordinate coord = new Coordinate(Math.abs(point.getValue() - 2048), absoluteDerivative);
            points.add(new LabelledPoint<>(point.getSensor(), point.getLabel(), point.getTimestamp(),
                    coord));
        }
        return points;
    }

    /**
     * Parse the given file to extract points. One file is expected to contain only one sensor.
     *
     * @param label state in which the sample was take (i.e. presence or absence)
     * @param file  path to the file
     * @return list of points in the file
     * @throws java.io.IOException
     */
    private static List<SensorValue> getPoints(String label, String file) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        InputStream resourceAsStream = Thread.currentThread().getContextClassLoader()
                .getResourceAsStream(file);
        MappingIterator<SensorValue> iterator = mapper.readValues(
                new JsonFactory().createParser(resourceAsStream),
                SensorValue.class);
        List<SensorValue> points = new ArrayList<>();
        while (iterator.hasNext()) {
            SensorValue next = iterator.next();
            next.setLabel(label);
            points.add(next);
        }
        return points;
    }

}
