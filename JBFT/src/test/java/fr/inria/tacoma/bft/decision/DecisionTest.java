package fr.inria.tacoma.bft.decision;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.frame.StateSet;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MassFunctionImpl;
import fr.inria.tacoma.bft.criteria.Criteria;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class DecisionTest {

    private FrameOfDiscernment frame;
    private DecisionStrategy decisionStrategy;

    @Before
    public void setUp() throws Exception {
        frame = FrameOfDiscernment
                .newFrame("position", "sitting", "standing", "layingDown");
        decisionStrategy = new CriteriaDecisionStrategy(0.4, 0.6, 0.8, Criteria::betP);
    }

    @Test
    public void DecisionMaking_WithVacuousMassFunction_ReturnsFullIgnoranceSetWith1AsConfidence() {
        MassFunction vacuousMassFunction = new MassFunctionImpl(frame);
        vacuousMassFunction.putRemainingOnIgnorance();
        Decision decision = decisionStrategy.decide(vacuousMassFunction);
        assertEquals(new Decision(frame.fullIgnoranceSet(), 1.0), decision);
    }

    @Test
    public void DecisionMaking_WithWholeMassOnOneState_ReturnGivenStateWith1AsConfidence() {
        MassFunction massFunction = new MassFunctionImpl(frame);
        StateSet sittingStateSet = frame.toStateSet("sitting");
        massFunction.set(sittingStateSet, 1.0);
        Decision decision = decisionStrategy.decide(massFunction);
        assertEquals(new Decision(sittingStateSet, 1.0), decision);
    }

    @Test
    public void DecisionMaking_WithMoreComplexMassFunction_ReturnsRightDecision() {
        MassFunction massFunction = new MassFunctionImpl(frame);
        massFunction.set(frame.toStateSet("sitting"), 0.6);
        massFunction.set(frame.toStateSet("standing"), 0.1);
        massFunction.putRemainingOnIgnorance();
        Decision decision = decisionStrategy.decide(massFunction);
        assertEquals(new Decision(frame.toStateSet("sitting"), 0.7), decision);
    }

    @Test
    public void DecisionMaking_WithLessSpecificMassFunction_ReturnsDecisionWithLessPreciseStateSet() {
        MassFunction massFunction = new MassFunctionImpl(frame);
        massFunction.set(frame.toStateSet("sitting"), 0.4);
        massFunction.set(frame.toStateSet("standing"), 0.2);
        massFunction.set(frame.toStateSet("standing", "layingDown"), 0.4);
        Decision decision = decisionStrategy.decide(massFunction);
        assertEquals(new Decision(frame.toStateSet("sitting", "standing"), 0.8), decision);
    }
}
