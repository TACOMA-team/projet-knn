package fr.inria.tacoma.bft.criteria;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MassFunctionImpl;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class CriteriaTest {


    private FrameOfDiscernment frame;
    private MassFunction massFunction;

    @Before
    public void setUp() {
        this.frame = FrameOfDiscernment.newFrame("position", "standing", "sitting", "layingDown");
        this.massFunction = new MassFunctionImpl(frame);
        this.massFunction.addToFocal(frame.toStateSet("sitting"), 0.2);
        this.massFunction.addToFocal(frame.toStateSet("sitting", "layingDown"), 0.3);
        this.massFunction.putRemainingOnIgnorance();
    }

    /*
     * Belief
     */
    @Test
    public void Belief_ForSetInFocalElements_ReturnsRightValue() {
        assertCriterionEquals(Criteria::belief, massFunction, 0.2, "sitting");
    }

    @Test
    public void Belief_ForSetInFocalElements2_ReturnRightValue() {
        assertCriterionEquals(Criteria::belief, massFunction, 0.5, "sitting", "layingDown");
    }

    @Test
    public void Belief_ForTotalIgnorance_ReturnsOne() {
        assertCriterionEquals(Criteria::belief, massFunction, 1.0, "standing", "sitting", "layingDown");
    }

    @Test
    public void Belief_ForNonFocalSet_ReturnsRightValue() {
        assertCriterionEquals(Criteria::belief, massFunction, 0.2, "sitting", "standing");
    }

    @Test
    public void Belief_WithMassOnVoid_ReturnsRightValue() {
        massFunction.set(frame.fullIgnoranceSet(), 0.0);
        massFunction.set(frame.emptyStateSet(), 0.2);
        massFunction.putRemainingOnIgnorance();
        assertCriterionEquals(Criteria::belief, massFunction, 0.2, "sitting");
    }

    /*
     * Plausibility
     */

    @Test
    public void Plausibility_ForFocalSet_ReturnsRightValue() {
        assertCriterionEquals(Criteria::plausibility, massFunction, 1.0, "sitting");
    }

    @Test
    public void Plausibility_ForFocalSet2_ReturnsRightValue() {
        assertCriterionEquals(Criteria::plausibility, massFunction, 1.0, "sitting", "layingDown");
    }

    @Test
    public void Plausibility_ForNonFocalSet_ReturnsRightValue() {
        assertCriterionEquals(Criteria::plausibility, massFunction, 1.0, "sitting", "standing");
    }

    @Test
    public void Plausibility_ForNonFocalSet2_ReturnsRightValue() {
        assertCriterionEquals(Criteria::plausibility, massFunction, 0.5, "standing");
    }

    /*
     * BetP
     */

    @Test
    public void BetP_ForFocalSet_ReturnsRightValue() {
        assertCriterionEquals(Criteria::betP, massFunction, 0.2 + 0.3/2 + 0.5/3, "sitting");
    }

    @Test
    public void BetP_ForFocalSet2_ReturnsRightValue() {
        assertCriterionEquals(Criteria::betP, massFunction, 0.2 + 0.3 + 0.5 * 2d/3d, "sitting", "layingDown");
    }

    @Test
    public void BetP_ForNonFocalSet_ReturnsRightValue() {
        assertCriterionEquals(Criteria::betP, massFunction, 0.2 + 0.3/2  + 0.5 * 2d/3d, "sitting", "standing");
    }

    @Test
    public void BetP_ForNonFocalSet2_ReturnsRightValue() {
        assertCriterionEquals(Criteria::betP, massFunction, 0.5/3, "standing");
    }


    private void assertCriterionEquals(Criterion criterion, MassFunction mass,
                                       double expected, String... states) {
        assertEquals(expected, criterion.apply(massFunction, frame.toStateSet(states)),
                MassFunctionImpl.PRECISION);
    }
}
