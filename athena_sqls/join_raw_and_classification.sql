SELECT uf.uid, u_email, uf.v_79, cf.classification, cf.confidence
FROM user_feedback as uf
JOIN classified_feedback as cf ON (cf.uid=uf.uid);