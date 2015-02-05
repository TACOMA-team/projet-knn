package fr.inria.tacoma.bft.combinations;

import fr.inria.tacoma.bft.core.frame.FrameOfDiscernment;
import fr.inria.tacoma.bft.core.mass.MassFunction;
import fr.inria.tacoma.bft.core.mass.MassFunctionImpl;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class DempsterCombinationTest {

	private BeliefCombination dempster;
	private FrameOfDiscernment frameOfDiscernment;
	private MassFunction mass1;
	private MassFunction mass2;
	private MassFunction mass3;
	private MassFunction mass1FusedWith2;
	private MassFunction mass1FusedWith3;

	@Before
	public void setup() {
		frameOfDiscernment = FrameOfDiscernment.newFrame("position",
                "standing", "sitting", "layingDown");
		mass1 = new MassFunctionImpl(frameOfDiscernment);
		mass2 = new MassFunctionImpl(frameOfDiscernment);
		mass3 = new MassFunctionImpl(frameOfDiscernment);
		mass1FusedWith2 = new MassFunctionImpl(frameOfDiscernment);
		mass1FusedWith3 = new MassFunctionImpl(frameOfDiscernment);

		mass1.addToFocal(frameOfDiscernment.toStateSet("layingDown"), 0.3);
        mass1.addToFocal(frameOfDiscernment.toStateSet("sitting", "layingDown"), 0.5);
        mass1.putRemainingOnIgnorance();

		mass2.addToFocal(frameOfDiscernment.toStateSet("sitting"), 0.6);
        mass2.putRemainingOnIgnorance();

		mass3.addToFocal(frameOfDiscernment.toStateSet("sitting", "standing"), 0.6);
        mass3.putRemainingOnIgnorance();

		mass1FusedWith2.addToFocal(frameOfDiscernment.toStateSet("sitting"), 0.5121951219512194);
        mass1FusedWith2.addToFocal(frameOfDiscernment.toStateSet("layingDown"), 0.14634146341463414);
        mass1FusedWith2.addToFocal(frameOfDiscernment.toStateSet("sitting", "layingDown"),
                        0.24390243902439024);
        mass1FusedWith2.putRemainingOnIgnorance();

		mass1FusedWith3.addToFocal(frameOfDiscernment.toStateSet("sitting"), 0.3658536585365853);
        mass1FusedWith3.addToFocal(frameOfDiscernment.toStateSet("layingDown"), 0.14634146341463414);
        mass1FusedWith3.addToFocal(frameOfDiscernment.toStateSet("sitting", "standing"),
                        0.14634146341463414);
        mass1FusedWith3.addToFocal(frameOfDiscernment.toStateSet("sitting", "layingDown"),
                        0.24390243902439024);
        mass1FusedWith3.putRemainingOnIgnorance();

		dempster = Combinations::dempster;
	}

	@Test
	public void DempsterCombination_Combine2Functions_ReturnsExpectedResult() {
		assertEquals(mass1FusedWith2, dempster.apply(mass2, mass1));
	}

	@Test
	public void DempsterCombination_CombinesTwoOtherFunctions_ReturnsExpectedResult() {
		assertEquals(mass1FusedWith3, dempster.apply(mass1, mass3));
	}

	@Test
	public void DempsterCombination_combineTwoFunctions_IsCommutative() {
		assertEquals("Dempster combination should be commutative.",
				dempster.apply(mass2, mass1), dempster.apply(mass1, mass2));
	}

	@Test
	public void DempsterCombination_combineTwoFunctions_IsAssociative() {
		assertEquals("Dempster combination should be associative.",
				dempster.apply(dempster.apply(mass1, mass2), mass3),
				dempster.apply(mass1,dempster.apply(mass2, mass3)));
	}

	@Test(expected = IllegalArgumentException.class)
	public void DempsterCombination_combineTwoFunctionsWithFullConflict_ThrowsException() {
		MassFunction conflictingFunction1 = new MassFunctionImpl(frameOfDiscernment);
		conflictingFunction1.addToFocal(frameOfDiscernment.toStateSet("sitting"), 1.0);
		MassFunction conflictingFunction2 = new MassFunctionImpl(frameOfDiscernment);
		conflictingFunction2.addToFocal(frameOfDiscernment.toStateSet("standing"), 1.0);
		dempster.apply(conflictingFunction1, conflictingFunction2);
	}
}
