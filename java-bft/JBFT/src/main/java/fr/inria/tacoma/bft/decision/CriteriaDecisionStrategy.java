package fr.inria.tacoma.bft.decision;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.criteria.Criteria;
import fr.inria.tacoma.bft.criteria.Criterion;

import java.util.Set;

/**
 * This class implements a decision strategy which computes a decision based on
 * several criteria. This algorithm aims to find a tradeoff between precision of
 * the state set and the validity of the decision. For instance, we prefer to
 * have a decision which says that the current state of our system is {A,B} with
 * 70% confidence than just {A} with 30% confidence.
 *
 * A state set must have every criteria (belief, betP, plausibility) higher than
 * a given thresholds and be the maximum for a given criteria to be chosen.
 * The algorithm starts by searching in state sets with 1 as cardinal and will
 * then increment the cardinal if no matching state set is found.
 *
 */
public class CriteriaDecisionStrategy implements DecisionStrategy {

    private final double beliefThreshold;
    private final double betPThreshold;
    private final double plausibilityThreshold;
    private final Criterion confidenceFun;

    /**
     * Creates a new CriteriaDecisionStrategy. This constructor takes the three
     * thresholds and which criterion to use to compute the confidence percentage
     * in the Decision.
     * @param beliefThreshold minimum for the belief
     * @param betPThreshold minimum for the betP
     * @param PlausibilityThreshold minimum for the plausibility
     * @param confidenceFun function which will be used to compute the confidence
     */
    public CriteriaDecisionStrategy(double beliefThreshold, double betPThreshold,
                                   double PlausibilityThreshold, Criterion confidenceFun) {
        this.beliefThreshold = beliefThreshold;
        this.betPThreshold = betPThreshold;
        this.plausibilityThreshold = PlausibilityThreshold;
        this.confidenceFun = confidenceFun;
    }

    @Override
    public Decision decide(MassFunction massFunction) {
        FrameOfDiscernment frame = massFunction.getFrameOfDiscernment();

        for (int card = 1; card < frame.card(); card++) {
            Set<StateSet> combinations = frame.getStateSetsWithCard(card);
            Decision decision =
                    findStateSetsWithMostConfidence(massFunction, combinations);
            if(decision != null && stateSetMatchesFilter(massFunction, decision.getStateSet())) {
                return decision;
            }
        }

        return new Decision(frame.fullIgnoranceSet(), 1.0);
    }

    /**
     * Checks that a StateSet matches the filter for this decision strategy.
     * @param massFunction Mass function for we we calculate the criteria
     * @param stateSet state set to work wirh
     * @return true if the given StateSet matches every criteria
     */
    private boolean stateSetMatchesFilter(MassFunction massFunction, StateSet stateSet) {
        return Criteria.belief(massFunction, stateSet) >= this.beliefThreshold
                && Criteria.betP(massFunction, stateSet) >= this.betPThreshold
                && Criteria.plausibility(massFunction, stateSet) >= this.plausibilityThreshold;
    }

    /**
     * Finds a decision with is the max for the given criterion. The max must be
     * unique, otherwise the function returns null.
     * @param massFunction mass function for which we want to find the max
     * @param stateSets every StateSet we want to consider
     * @return the base decision in the given state sets
     */
    private Decision findStateSetsWithMostConfidence(MassFunction massFunction, Set<StateSet> stateSets) {
        double maxConfidence = 0;
        StateSet mostConfidentStateSet = null;
        for (StateSet stateSet : stateSets) {
            double confidence = this.confidenceFun.apply(massFunction, stateSet);
            if(confidence > maxConfidence) {
                maxConfidence = confidence;
                mostConfidentStateSet = stateSet;
            }
            else if(confidence == maxConfidence) {
                // We have several possible maximum, which mean we cannot take a decision
                mostConfidentStateSet = null;
            }
        }

        if (mostConfidentStateSet == null) {
            return null;
        } else {
            return new Decision(mostConfidentStateSet, maxConfidence);
        }
    }


}
