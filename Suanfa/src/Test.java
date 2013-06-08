public class Test {
	public static void Joseph (int low, int up) {
		int result= 0; // 求解计数，查看满足条件的解的个数
		int result1 = (int) (Math.log(up) / Math.log(2));
		int result2 = (int) (Math.log(low) / Math.log(2));			
		result =result1 - result2 + 1;
		System.out.println("在区间["+ low +","+ up + "]内满足要求解共有"+result+"个");
			}

	public static void main(String[] args) {
		
		Joseph(1, 2005);// 求解出解的个数
		
	}
}
