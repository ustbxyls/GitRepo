public class Test {
	public static void Joseph (int low, int up) {
		int result= 0; // ���������鿴���������Ľ�ĸ���
		int result1 = (int) (Math.log(up) / Math.log(2));
		int result2 = (int) (Math.log(low) / Math.log(2));			
		result =result1 - result2 + 1;
		System.out.println("������["+ low +","+ up + "]������Ҫ��⹲��"+result+"��");
			}

	public static void main(String[] args) {
		
		Joseph(1, 2005);// ������ĸ���
		
	}
}
