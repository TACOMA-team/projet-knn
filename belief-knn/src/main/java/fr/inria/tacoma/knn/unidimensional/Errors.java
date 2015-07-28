package fr.inria.tacoma.knn.unidimensional;

import fr.inria.tacoma.bft.combinations.Combinations;
import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.criteria.Criteria;
import fr.inria.tacoma.bft.decision.CriteriaDecisionStrategy;
import fr.inria.tacoma.bft.decision.Decision;
import fr.inria.tacoma.bft.decision.DecisionStrategy;
import fr.inria.tacoma.bft.sensorbelief.SensorBeliefModel;
import fr.inria.tacoma.knn.core.LabelledPoint;
import org.jfree.chart.ChartPanel;

import javax.swing.*;
import java.util.*;

public class Errors {

    public static void showErrors(SensorBeliefModel<Double> model,
                           List<LabelledPoint<Double>> crossValidation) {
        DecisionStrategy decisionStrategy =
                new CriteriaDecisionStrategy(0.5, 0.6, 0.7, Criteria::betP);
        int errorCount = 0;
        int imprecisionCount = 0;

        for (LabelledPoint<Double> sensorValue : crossValidation) {
            Decision decision = decisionStrategy.decide(
                    model.toMass(sensorValue.getValue()));
            StateSet actualDecision = decision.getStateSet();
            StateSet expectedDecision = model.getFrame().toStateSet(sensorValue.getLabel());

            if (!actualDecision.includesOrEquals(expectedDecision)) {
                //System.out.println("bad decision for value: " + sensorValue.getValue());
                errorCount++;
            }
            else if (!actualDecision.equals(expectedDecision)) {
                imprecisionCount++;
            }
        }
        System.out.println(errorCount + " errors out of " + crossValidation.size()
                + " (" + (double)errorCount * 100 / crossValidation.size() +" %) tested point" +
                " with decision algorithm.");
        System.out.println(imprecisionCount + " imprecision out of " + crossValidation.size()
                + " (" + (double)imprecisionCount * 100 / crossValidation.size() +" %) tested point" +
                " with decision algorithm.");
    }

    /**
     *
     * @param models map from sensor model to their sample
     */
    public static void showGlobalErrors(
            Map<SensorBeliefModel<Double>,List<LabelledPoint<Double>>> models) {
        DecisionStrategy decisionStrategy =
                new CriteriaDecisionStrategy(0.5, 0.6, 0.7, Criteria::betP);
        int errorCount = 0;
        int imprecisionCount = 0;
        int fullIgnoranceCount = 0;

        List<LabelledPoint<Double>> referenceList = models.values().iterator().next();
        int size = referenceList.size();
        FrameOfDiscernment frame = referenceList.get(0).getStateSet().getFrame();


        for (int i = 0; i < size; i++) {
            final int index = i;
            MassFunction resultingMass = models.entrySet().stream()
                    .map(entry -> entry.getKey().toMass(entry.getValue().get(index).getValue()))
                    .reduce(Combinations::dempster).get();

            Decision decision = decisionStrategy.decide(resultingMass);
            StateSet actualDecision = decision.getStateSet();

            if (!actualDecision.includesOrEquals(referenceList.get(i).getStateSet())) {
                errorCount++;
            }
            else if (actualDecision.equals(frame.fullIgnoranceSet())) {
                fullIgnoranceCount++;
            }
            else if (!actualDecision.equals(referenceList.get(i).getStateSet())) {
                imprecisionCount++;
            }
        }

        System.out.println(errorCount + " errors out of " + size
                + " (" + (double)errorCount * 100 / size +" %) tested point" +
                " with decision algorithm.");
        System.out.println(fullIgnoranceCount + " full ignorance out of " + size
                + " (" + (double)fullIgnoranceCount * 100 / size +" %) tested point" +
                " with decision algorithm.");
        System.out.println(imprecisionCount + " imprecision out of " + size
                + " (" + (double)imprecisionCount * 100 / size +" %) tested point" +
                " with decision algorithm.");
    }

    public static Map<Double, MassFunction> showTimeLine(
            Map<SensorBeliefModel<Double>,List<LabelledPoint<Double>>> models) {
        List<LabelledPoint<Double>> referenceList = models.values().iterator().next();
        int size = referenceList.size();
        FrameOfDiscernment frame = referenceList.get(0).getStateSet().getFrame();
        SortedMap<Double, MassFunction> timeline = new TreeMap<>();

        for (int i = 0; i < size; i++) {
            final int index = i;
            MassFunction resultingMass = models.entrySet().stream()
                    .map(entry -> entry.getKey().toMass(entry.getValue().get(index).getValue()))
                    .reduce(Combinations::dempster).get();
            timeline.put((double)i, resultingMass);
        }

        ChartPanel chartPanel = JfreeChartDisplay1D.getChartPanel(timeline, frame);
        JFrame windowFrame = new JFrame();
        windowFrame.setTitle("timeline");
        windowFrame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        windowFrame.setContentPane(chartPanel);
        windowFrame.pack();
        windowFrame.setVisible(true);

        return timeline;
    }
}
